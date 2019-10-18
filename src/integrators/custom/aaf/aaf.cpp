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

#include <mitsuba/mitsuba.h>
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

 struct PrimaryRayData {
	Intersection *its;
	RayDifferential *primaryRay;
    int objectId;
 };

 // Ideally one would adaptively sampling each source and run the filter in image space for each source seperately
 // One optimization would be combine similar size filters into one on per pixel basis.
 struct PerPixelData {
	 Float *d1;
	 Float *d2Max; // Use for primal filter size 
	 Float *d2Min; // Use for computing adaptive sampling rates

	 Float *beta; // primal filter size
	 Spectrum *colorEmitter; // color for each emiiter
	 Spectrum color; // color for direct hit to light source

	 void init(size_t nEmitters) {
		 d1 = new Float[nEmitters];
		 d2Max = new Float[nEmitters];
		 d2Min = new Float[nEmitters];

		 beta = new Float[nEmitters];
		 colorEmitter = new Spectrum[nEmitters];

		 for (size_t i = 0; i < nEmitters; i++) {
			 beta[i] = 1;
			 d1[i] = 0;
			 d2Max[i] = std::numeric_limits<Float>::min();
			 d2Min[i] = std::numeric_limits<Float>::max();
			 colorEmitter[i] = Spectrum(0.0f);
		 }

		 color = Spectrum(0.0f);
	 }

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

	void collectEmitterParameters(Scene *scene, std::vector<Point> &emitterCenter) {
		for (auto emitter : scene->getEmitters()) {
			if (emitter->isOnSurface() &&
				emitter->getShape() != NULL) {
				if (emitter->getShape()->getName().compare("rectangle") == 0) {
					Point c = emitter->getShape()->getCenter();
					emitterCenter.push_back(c);
				}
				else
					std::cerr << "AAF: Ignoring emitter geometry other than Rectangle/Parallelogram." << std::endl;
			}
		}
	}

    bool render(Scene *scene,
                RenderQueue *queue, const RenderJob *job,
                int sceneResID, int sensorResID, int samplerResID) {

		ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();
        auto cropSize = film->getCropSize();
		size_t nCores = sched->getCoreCount();
        Sampler *sampler_main = static_cast<Sampler *>(sched->getResource(samplerResID, 0));
        size_t sampleCount = sampler_main->getSampleCount();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
                " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
            nCores == 1 ? "core" : "cores");

		// global buffer
		PrimaryRayData *gBuffer = new PrimaryRayData[cropSize.x * cropSize.y * sampleCount];
		std::vector<Point> emitterCenters;
		PerPixelData *ppd = new PerPixelData[cropSize.x * cropSize.y];

		collectEmitterParameters(scene, emitterCenters);
				
        // Results for saving the computation
        ref<Bitmap> result = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
        result->clear();

        struct ThreadData {
            ref<Sampler> sampler;
        };

        std::vector<ThreadData> threadData;
        for(auto i = 0; i < nCores; i++) {
            threadData.emplace_back(ThreadData {
                    sampler_main->clone()
            });
        }

        BlockScheduler runPool(cropSize.x * cropSize.y, (int)nCores, 1);
        runPool.run([&](int pixelID, int threadID) {
			ThreadData &td = threadData[threadID];
			auto sampler = td.sampler.get();

            bool needsApertureSample = sensor->needsApertureSample();
            bool needsTimeSample = sensor->needsTimeSample();

            RadianceQueryRecord rRec(scene, sampler);
            Point2 apertureSample(0.5f);
            Float timeSample = 0.5f;
            RayDifferential sensorRay;
            uint32_t queryType = RadianceQueryRecord::ESensorRay;

			int i = pixelID % cropSize.x;
			int j = pixelID / cropSize.x;

            Point2i offset = Point2i(i, j);
			sampler->generate(offset);
			for (size_t k = 0; k < sampleCount; k++) {
				rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
					apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                sensor->sampleRayDifferential(
					sensorRay, samplePos, apertureSample, timeSample);
				gBufferPass(sensorRay, rRec, gBuffer[pixelID *  sampleCount + k]);
                sampler->advance();
            }

			ppd[pixelID].init(emitterCenters.size());
        });

		std::cout << "Finished GBuffer pass." << std::endl;

		runPool.run([&](int pixelID, int threadID) {
			ThreadData &td = threadData[threadID];
			auto sampler = td.sampler.get();
			RadianceQueryRecord rRec(scene, sampler);
			
			int i = pixelID % cropSize.x;
			int j = pixelID / cropSize.x;
			sampler->generate(Point2i(i, j));

			for (size_t k = 0; k < sampleCount; k++) {
				collectSamples(rRec, gBuffer[pixelID *  sampleCount + k], ppd[pixelID], emitterCenters);
				sampler->advance();
			}

			for (size_t k = 0; k < emitterCenters.size(); k++)
				ppd[pixelID].colorEmitter[k] /= (float)sampleCount;

			ppd[pixelID].color /= (float)sampleCount;
		});

		std::cout << "Finished Data-collection and adaptive sampling pass." << std::endl;

		pBufferToImage(result, ppd, cropSize, (int)emitterCenters.size());
		//gBufferToImage(result, gBuffer, cropSize, sampleCount);
        film->setBitmap(result);
		
		return true;
    }

	 void gBufferPass(const RayDifferential &primaryRay, RadianceQueryRecord &rRec, PrimaryRayData &prd) {
        Intersection &its = rRec.its;
		RayDifferential ray(primaryRay);
        bool intersect = rRec.rayIntersect(ray);
		ray.mint = Epsilon;

		prd.primaryRay = new RayDifferential(primaryRay);
		prd.its = new Intersection(rRec.its);
		
		if (!intersect) {
			prd.objectId = -2;
			return;
		}
		else if (intersect && its.isEmitter()) {
			prd.objectId = -1;
			return;
		}
		      
		prd.objectId = 0;
    }

	void collectSamples(RadianceQueryRecord &rRec, const PrimaryRayData &prd, PerPixelData &ppd, const std::vector<Point> &emitterCenters) {
		if (prd.objectId == -2)
			return;
		else if (prd.objectId == -1) {
			ppd.color += prd.its->Le(-prd.primaryRay->d);
			return;
		}
				
		size_t nEmitterSamples = 9;

		DirectSamplingRecord dRec(*prd.its);
		
		const Scene *scene = rRec.scene;
		Intersection its;
		
		int emitterIdx = 0;
		for (auto emitter : scene->getEmitters()) {
			if (emitter->isOnSurface() &&
				emitter->getShape() != NULL &&
				emitter->getShape()->getName().compare("rectangle") == 0) {

				// This implementation doesn't look right
				// Read soler et. al 1998 for r(x) term
				// read egan 11a and 11b
				Spectrum unshadowedIrradiance = getUnshadowedIrradiance(emitter, emitterCenters[emitterIdx], *prd.its, prd.its->getBSDF(*prd.primaryRay));
				Float hitCount = 0;
				for (size_t i = 0; i < nEmitterSamples; i++) {
					Spectrum value = emitter->sampleDirect(dRec, rRec.nextSample2D());
					RayDifferential shadowRay(prd.its->p, dRec.d, prd.primaryRay->time);
					bool intersect = scene->rayIntersect(shadowRay, its);
					hitCount += intersect && its.isEmitter() ? 1 : 0;
				}

				ppd.colorEmitter[emitterIdx] += unshadowedIrradiance * Spectrum(hitCount / (Float)nEmitterSamples);

				emitterIdx++;
			}

		}
	}

	Spectrum getUnshadowedIrradiance(const Emitter *emitter, const Point &emitterHit, const Intersection &receiverIts, const BSDF *bsdf) {
		Vector emitterDirection = emitterHit - receiverIts.p;
		Float r_2 = dot(emitterDirection, emitterDirection);
		Float r = sqrt(r_2);
		emitterDirection /= r;
		Frame emitterFrame = emitter->getShape()->getFrame();
		Float cosEmitter = dot(emitterFrame.n, emitterDirection);
		if (cosEmitter >= 0)
			return Spectrum(0.0f);

		BSDFSamplingRecord bRec(receiverIts, receiverIts.toLocal(emitterDirection));
		Spectrum radiance = emitter->eval(receiverIts, emitterDirection);
		
		return radiance * bsdf->eval(bRec) * abs(cosEmitter) / r_2;
	}
   
	void gBufferToImage(ref<Bitmap> &result, const PrimaryRayData *gBuffer, const Vector2i &cropSize, const size_t sampleCount) {
		int select = 1;

		Spectrum *throughputPix = (Spectrum *)result->getData();
		Spectrum value(0.0f);

		for (size_t j = 0; j < cropSize.y; j++)
			for (size_t i = 0; i < cropSize.x; i++) {
				size_t currPix = j * cropSize.x + i;

				for (size_t k = 0; k < sampleCount; k++) {
					const PrimaryRayData &prd = gBuffer[currPix *  sampleCount + k];
					if (select == 0)
						value.fromLinearRGB(prd.its->shFrame.n.x, prd.its->shFrame.n.y, prd.its->shFrame.n.z);
					else if (select == 1)
						value = prd.its->getBSDF(*prd.primaryRay)->getDiffuseReflectance(*prd.its);
					else if (select == 2)
						value = prd.its->getBSDF(*prd.primaryRay)->getSpecularReflectance(*prd.its);
					else if (select == 3)
						value = prd.its->color; // vertex interpolated color
					else
						std::cerr << "gNufferToImage:Undefined choice." << std::endl;

					throughputPix[currPix] += value;
				}
			}
		result->scale(1.0f / sampleCount);
	}

	void pBufferToImage(ref<Bitmap> &result, const PerPixelData *pBuffer, const Vector2i &cropSize, const int nEmitters) {
		Spectrum *throughputPix = (Spectrum *)result->getData();
		
		for (size_t j = 0; j < cropSize.y; j++)
			for (size_t i = 0; i < cropSize.x; i++) {
				size_t currPix = j * cropSize.x + i;
				const PerPixelData &pData = pBuffer[currPix];
				throughputPix[currPix] = pData.color;

				for (int k = 0; k < nEmitters; k++) {
					throughputPix[currPix] += pData.colorEmitter[k];
				}
			}
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
