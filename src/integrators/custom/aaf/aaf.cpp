/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/integrator.h>

#include <functional>
#include <algorithm>
#include <thread>

#include <mitsuba/core/lock.h>
#include <mitsuba/core/thread.h>

MTS_NAMESPACE_BEGIN

/************************************************************************
   * Block Scheduler
   ************************************************************************/
class BlockScheduler {
    /************************************************************************
         * Local Class - Block Thread
         ************************************************************************/
public:

    class BlockThread : public Thread {
    public:
        //Store the lambda
        typedef std::function<void(int, int)> ComputeBlockFunction;
        ComputeBlockFunction localF;
        int localTID = -1;
        BlockScheduler &localParent;

        //Constructor
        BlockThread(const std::string &threadName, ComputeBlockFunction f, int tid, BlockScheduler &parent)
                : Thread(threadName), localParent(parent) {
            this->localF = f;
            this->localTID = tid;
            this->setPriority(EThreadPriority::ENormalPriority);
        }

        void run() override {
            while (true) {
                auto blockIdx = localParent.getBlockIdx();
                if (blockIdx.first < 0) break;
                for (int i = blockIdx.first; i < blockIdx.second; i++) {
                    localF(i, localTID);
                }
            }
        }
    };


    /************************************************************************
         * Constructor
         ************************************************************************/
public:

    BlockScheduler(int numBlocks, int numThreads, int taskGroup) :
            numBlocks(numBlocks), numThreads(numThreads), blockIdx(0), task_group(taskGroup) {
        mutex = new Mutex();
    }


    /************************************************************************
         * Typedefs - ComputeBlockFunction
         ************************************************************************/
public:

    typedef std::function<void(int, int)> ComputeBlockFunction;


    /************************************************************************
         * Public Functions
         ************************************************************************/
public:

    /**
     * Runs a ComputeBlockFunctino for numBlocks on numThreads
     */
    void run(ComputeBlockFunction f) {
        blockIdx = 0;

        ref_vector<BlockThread> group;
        for (int tid = 0; tid < numThreads; ++tid) {
            ref<BlockThread> bThread = new BlockThread("BLOCKTHREAD" + std::to_string(tid), f, tid, *this);
            group.push_back(bThread);
            bThread->start();
        }

        for (int tid = 0; tid < numThreads; ++tid) {
            group[tid]->join();
        }
    }

    /**
         * Return a unique block for each thread.
         * Return a negative number when no blocks are available.
         */
    std::pair<int, int> getBlockIdx() {
        LockGuard lock(mutex);
        if (blockIdx >= numBlocks) return std::pair<int, int>(-1, -1);

        int v = blockIdx;
        int vMax = std::min(blockIdx + task_group, numBlocks);
        blockIdx += task_group;
        return std::pair<int, int>(v, vMax);
    }


    /************************************************************************
         * Destructor
         ************************************************************************/
public:

    ~BlockScheduler() = default;


    /************************************************************************
       * Private Class Variables
       ************************************************************************/
private:
    int task_group;
    int numBlocks;
    int numThreads;
    int blockIdx;
    ref<Mutex> mutex;
};

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

/*! \plugin{path}{Path tracer}
 * \order{2}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
 *	   }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See the description below
 *        for details.\default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 *
 * This integrator implements a basic path tracer and is a \emph{good default choice}
 * when there is no strong reason to prefer another method.
 *
 * To use the path tracer appropriately, it is instructive to know roughly how
 * it works: its main operation is to trace many light paths using \emph{random walks}
 * starting from the sensor. A single random walk is shown below, which entails
 * casting a ray associated with a pixel in the output image and searching for
 * the first visible intersection. A new direction is then chosen at the intersection,
 * and the ray-casting step repeats over and over again (until one of several
 * stopping criteria applies).
 * \begin{center}
 * \includegraphics[width=.7\textwidth]{images/integrator_path_figure.pdf}
 * \end{center}
 * At every intersection, the path tracer tries to create a connection to
 * the light source in an attempt to find a \emph{complete} path along which
 * light can flow from the emitter to the sensor. This of course only works
 * when there is no occluding object between the intersection and the emitter.
 *
 * This directly translates into a category of scenes where
 * a path tracer can be expected to produce reasonable results: this is the case
 * when the emitters are easily ``accessible'' by the contents of the scene. For instance,
 * an interior scene that is lit by an area light will be considerably harder
 * to render when this area light is inside a glass enclosure (which
 * effectively counts as an occluder).
 *
 * Like the \pluginref{direct} plugin, the path tracer internally relies on multiple importance
 * sampling to combine BSDF and emitter samples. The main difference in comparison
 * to the former plugin is that it considers light paths of arbitrary length to compute
 * both direct and indirect illumination.
 *
 * For good results, combine the path tracer with one of the
 * low-discrepancy sample generators (i.e. \pluginref{ldsampler},
 * \pluginref{halton}, or \pluginref{sobol}).
 *
 * \paragraph{Strict normals:}\label{sec:strictnormals}
 * Triangle meshes often rely on interpolated shading normals
 * to suppress the inherently faceted appearance of the underlying geometry. These
 * ``fake'' normals are not without problems, however. They can lead to paradoxical
 * situations where a light ray impinges on an object from a direction that is classified as ``outside''
 * according to the shading normal, and ``inside'' according to the true geometric normal.
 *
 * The \code{strictNormals}
 * parameter specifies the intended behavior when such cases arise. The default (\code{false}, i.e. ``carry on'')
 * gives precedence to information given by the shading normal and considers such light paths to be valid.
 * This can theoretically cause light ``leaks'' through boundaries, but it is not much of a problem in practice.
 *
 * When set to \code{true}, the path tracer detects inconsistencies and ignores these paths. When objects
 * are poorly tesselated, this latter option may cause them to lose a significant amount of the incident
 * radiation (or, in other words, they will look dark).
 *
 * The bidirectional integrators in Mitsuba (\pluginref{bdpt}, \pluginref{pssmlt}, \pluginref{mlt} ...)
 * implicitly have \code{strictNormals} set to \code{true}. Hence, another use of this parameter
 * is to match renderings created by these methods.
 *
 * \remarks{
 *    \item This integrator does not handle participating media
 *    \item This integrator has poor convergence properties when rendering
 *    caustics and similar effects. In this case, \pluginref{bdpt} or
 *    one of the photon mappers may be preferable.
 * }
 */

 struct PerRayData {
    Color3 color;
    Normal normal;
    Point3f hit;
    int objectId;
 };

class AAF : public Integrator {
public:
	AAF(const Properties &props)
		: Integrator(props) {

        m_rrDepth = props.getInteger("rrDepth", 5);

        /* Longest visualized path depth (\c -1 = infinite).
           A value of \c 1 will visualize only directly visible light sources.
           \c 2 will lead to single-bounce (direct-only) illumination, and so on. */
        m_maxDepth = props.getInteger("maxDepth", -1);

        /**
         * This parameter specifies the action to be taken when the geometric
         * and shading normals of a surface don't agree on whether a ray is on
         * the front or back-side of a surface.
         *
         * When \c strictNormals is set to \c false, the shading normal has
         * precedence, and rendering proceeds normally at the risk of
         * introducing small light leaks (this is the default).
         *
         * When \c strictNormals is set to \c true, the random walk is
         * terminated when encountering such a situation. This may
         * lead to silhouette darkening on badly tesselated meshes.
         */
        m_strictNormals = props.getBoolean("strictNormals", false);

        /**
         * When this flag is set to true, contributions from directly
         * visible emitters will not be included in the rendered image
         */
        m_hideEmitters = props.getBoolean("hideEmitters", false);

        if (m_rrDepth <= 0)
            Log(EError, "'rrDepth' must be set to a value greater than zero!");

        if (m_maxDepth <= 0 && m_maxDepth != -1)
            Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");

		m_minDepth = props.getInteger("minDepth", 0);
		if(m_maxDepth != -1 && m_minDepth+1 >= m_maxDepth) {
			Log(EError, "Impossible to have the good min depth");
		}

	}

	/// Unserialize from a binary data stream
	AAF(Stream *stream, InstanceManager *manager)
		: Integrator(stream, manager) { }

	bool preprocess(const Scene *scene, RenderQueue *queue,
									const RenderJob *job, int sceneResID, int sensorResID,
									int samplerResID) override {
		Integrator::preprocess(scene, queue, job, sceneResID,
													 sensorResID, samplerResID);
		return true;
	}

    virtual void cancel() {

	}

    bool render(Scene *scene,
                RenderQueue *queue, const RenderJob *job,
                int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();
        auto cropSize = film->getCropSize();
        size_t nCores = 1; //sched->getCoreCount();
        Sampler *sampler_main = static_cast<Sampler *>(sched->getResource(samplerResID, 0));
        size_t sampleCount = sampler_main->getSampleCount();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
                " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
            nCores == 1 ? "core" : "cores");

        // Results for saving the computation
        ref<Bitmap> result = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
        result->clear();
        struct ThreadData {
            ref<Sampler> sampler;
            ref<Bitmap> image;
            PerRayData *prd;
        };
        std::vector<ThreadData> threadData;
        for(auto i = 0; i < nCores; i++) {
            threadData.emplace_back(ThreadData {
                    sampler_main->clone(),
                    result->clone(),
                    new PerRayData[cropSize.x * cropSize.y]
            });
        }

        BlockScheduler runPool(nCores, nCores, 1);
        runPool.run([&](int tileID, int threadID) {
            ThreadData& data = threadData[threadID];
            auto sampler = data.sampler.get();

            auto accumulate = [&](const Point2& pixel, const PerRayData& value) {
                Spectrum *throughputPix = (Spectrum *) data.image->getData();
                size_t curr_pix = ((int)pixel.y) * cropSize.x + ((int)pixel.x);
                throughputPix[curr_pix].fromLinearRGB(value.normal.x, value.normal.y, value.normal.z);
            };

            // Other things from MTS
            Float diffScaleFactor = 1.0f /
                                    std::sqrt((Float) sampler->getSampleCount());

            bool needsApertureSample = sensor->needsApertureSample();
            bool needsTimeSample = sensor->needsTimeSample();

            RadianceQueryRecord rRec(scene, sampler);
            Point2 apertureSample(0.5f);
            Float timeSample = 0.5f;
            RayDifferential sensorRay;
            uint32_t queryType = RadianceQueryRecord::ESensorRay;

//            if (!sensor->getFilm()->hasAlpha()) /* Don't compute an alpha channel if we don't have to */
//                queryType &= ~RadianceQueryRecord::EOpacity;

            for (size_t i = 0; i < cropSize.x; ++i) {
                for (size_t j = 0; j < cropSize.y; ++j) {
                    Point2i offset = Point2i(i, j);
                    sampler->generate(offset);

                    //for (size_t k = 0; k < sampler->getSampleCount(); k++) {
                    rRec.newQuery(queryType, sensor->getMedium());
                    Point2 samplePos(Point2(offset) + Vector2(0.5f, 0.5f));

                    if (needsApertureSample)
                        apertureSample = rRec.nextSample2D();
                    if (needsTimeSample)
                        timeSample = rRec.nextSample1D();

                    Spectrum spec = sensor->sampleRayDifferential(
                            sensorRay, samplePos, apertureSample, timeSample);

                    sensorRay.scaleDifferential(diffScaleFactor);

                    gBufferPass(sensorRay, rRec, data.prd[(int)samplePos.y * cropSize.x + (int)samplePos.x]);
                    //spec *= Li(sensorRay, rRec);
                    accumulate(samplePos, data.prd[(int)samplePos.y * cropSize.x + (int)samplePos.x]);
                    sampler->advance();
                    //}
                }
            }
        });

        // Save the results
        for(auto res: threadData) {
            result->accumulate(res.image);
        }
        result->scale(1.0 / nCores); // Average across the different cores
        film->setBitmap(result);


        return true;
    }

    void gBufferPass(const RayDifferential &primaryRay, RadianceQueryRecord &rRec, PerRayData &prd) {
        //const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		RayDifferential ray(primaryRay);
        rRec.rayIntersect(ray);
		ray.mint = Epsilon;

        prd.normal = its.shFrame.n;
        prd.color = its.color;
    }
    

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		RayDifferential ray(r);
		ray.hasDifferentials = false; // To be consistent with BDPT
		Spectrum Li(0.0f);
		bool scattered = false;

		/* Perform the first ray intersection (or ignore if the
		   intersection has already been provided). */
		rRec.rayIntersect(ray);
		ray.mint = Epsilon;

		Spectrum throughput(1.0f);
		Float eta = 1.0f;

		while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
			if (!its.isValid()) {
				/* If no intersection could be found, potentially return
				   radiance from a environment luminaire if it exists */
				if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
					&& (!m_hideEmitters || scattered))
					Li += throughput * scene->evalEnvironment(ray);
				break;
			}

			const BSDF *bsdf = its.getBSDF(ray);

			/* Possibly include emitted radiance if requested */
			if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
				&& (!m_hideEmitters || scattered)
				&& rRec.depth > m_minDepth)
				Li += throughput * its.Le(-ray.d);

			/* Include radiance from a subsurface scattering model if requested */
			if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
				Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

			if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
				|| (m_strictNormals && dot(ray.d, its.geoFrame.n)
					* Frame::cosTheta(its.wi) >= 0)) {

				/* Only continue if:
				   1. The current path length is below the specifed maximum
				   2. If 'strictNormals'=true, when the geometric and shading
				      normals classify the incident direction to the same side */
				break;
			}

			/* ==================================================================== */
			/*                     Direct illumination sampling                     */
			/* ==================================================================== */

			/* Estimate the direct illumination if this is requested */
			DirectSamplingRecord dRec(its);

			if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
				(bsdf->getType() & BSDF::ESmooth) && rRec.depth > m_minDepth) {
				Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
				if (!value.isZero()) {
					const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

					/* Allocate a record for querying the BSDF */
					BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

					/* Evaluate BSDF * cos(theta) */
					const Spectrum bsdfVal = bsdf->eval(bRec);

					/* Prevent light leaks due to the use of shading normals */
					if (!bsdfVal.isZero() && (!m_strictNormals
							|| dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

						/* Calculate prob. of having generated that direction
						   using BSDF sampling */
						Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
							? bsdf->pdf(bRec) : 0;

						/* Weight using the power heuristic */
						Float weight = miWeight(dRec.pdf, bsdfPdf);
						Li += throughput * value * bsdfVal * weight;
					}
				}
			}

			/* ==================================================================== */
			/*                            BSDF sampling                             */
			/* ==================================================================== */

			/* Sample BSDF * cos(theta) */
			Float bsdfPdf;
			BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
			Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
			if (bsdfWeight.isZero())
				break;

			scattered |= bRec.sampledType != BSDF::ENull;

			/* Prevent light leaks due to the use of shading normals */
			const Vector wo = its.toWorld(bRec.wo);
			Float woDotGeoN = dot(its.geoFrame.n, wo);
			if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
				break;

			bool hitEmitter = false;
			Spectrum value;

			/* Trace a ray in this direction */
			ray = Ray(its.p, wo, ray.time);
			if (scene->rayIntersect(ray, its)) {
				/* Intersected something - check if it was a luminaire */
				if (its.isEmitter()) {
					value = its.Le(-ray.d);
					dRec.setQuery(ray, its);
					hitEmitter = true;
				}
			} else {
				/* Intersected nothing -- perhaps there is an environment map? */
				const Emitter *env = scene->getEnvironmentEmitter();

				if (env) {
					if (m_hideEmitters && !scattered)
						break;

					value = env->evalEnvironment(ray);
					if (!env->fillDirectSamplingRecord(dRec, ray))
						break;
					hitEmitter = true;
				} else {
					break;
				}
			}

			/* Keep track of the throughput and relative
			   refractive index along the path */
			throughput *= bsdfWeight;
			eta *= bRec.eta;

			/* If a luminaire was hit, estimate the local illumination and
			   weight using the power heuristic */
			if (hitEmitter &&
				(rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)
					&& rRec.depth > m_minDepth) {
				/* Compute the prob. of generating that direction using the
				   implemented direct illumination sampling technique */
				const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
					scene->pdfEmitterDirect(dRec) : 0;
				Li += throughput * value * miWeight(bsdfPdf, lumPdf);
			}

			/* ==================================================================== */
			/*                         Indirect illumination                        */
			/* ==================================================================== */

			/* Set the recursive query type. Stop if no surface was hit by the
			   BSDF sample or if indirect illumination was not requested */
			if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
				break;
			rRec.type = RadianceQueryRecord::ERadianceNoEmission;

			if (rRec.depth++ >= m_rrDepth) {
				/* Russian roulette: try to keep path weights equal to one,
				   while accounting for the solid angle compression at refractive
				   index boundaries. Stop with at least some probability to avoid
				   getting stuck (e.g. due to total internal reflection) */

				Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
				if (rRec.nextSample1D() >= q)
					break;
				throughput /= q;
			}
		}

		/* Store statistics */
		avgPathLength.incrementBase();
		avgPathLength += rRec.depth;

		return Li;
	}

	inline Float miWeight(Float pdfA, Float pdfB) const {
		pdfA *= pdfA;
		pdfB *= pdfB;
		return pdfA / (pdfA + pdfB);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "AAF[" << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rrDepth = " << m_rrDepth << "," << endl
			<< "  strictNormals = " << m_strictNormals << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()

private:
    int m_minDepth;

    int m_maxDepth;
    int m_rrDepth;
    bool m_strictNormals;
    bool m_hideEmitters;

};

MTS_IMPLEMENT_CLASS_S(AAF, false, Integrator)
MTS_EXPORT_PLUGIN(AAF, "AAF");
MTS_NAMESPACE_END
