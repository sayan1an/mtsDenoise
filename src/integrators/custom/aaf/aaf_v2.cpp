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

 struct PrimaryRayData 
 {
	Intersection *its;
	RayDifferential *primaryRay;
	Float depth;
    int objectId;
 };

#define GET_MAT3x3_IDENTITY ([](Matrix3x3 &m) {	m.setIdentity(); return static_cast<const Matrix3x3>(m); } (Matrix3x3(0.0f)))

 struct TriangleEmitters
 {
	 uint32_t idx[3];
	 const Point *vertexPositions; // pointer to world positions
	
	 void init(const Point *vertices, uint32_t i0, uint32_t i1, uint32_t i2)
	 {
		 vertexPositions = vertices;
		 idx[0] = i0;
		 idx[1] = i1;
		 idx[2] = i2;
	 }

	 Normal computeAreaNormal(Float &area, const Matrix3x3 &w2l = GET_MAT3x3_IDENTITY) const
	 {	
		 const Point p0 = Point(w2l * Vector(vertexPositions[idx[0]]));
		 const Point p1 = Point(w2l * Vector(vertexPositions[idx[1]]));
		 const Point p2 = Point(w2l * Vector(vertexPositions[idx[2]]));

		 return computeAreaNormal(area, p0, p1, p2);
	 }
		
	 // return pdf in solid angle
	 // sample in local space
	 // multiply the pdf with appropriate jacobian
	 Float sample(Point &p, Point2f &sample, const Point &ref = Point(0.0f), 
			const Matrix3x3 &w2l = GET_MAT3x3_IDENTITY,
			Float w2lDet = 1.0f,
			const Matrix3x3 &l2w = GET_MAT3x3_IDENTITY) const
	 {	
		 const Point p0 = Point(w2l * (vertexPositions[idx[0]] - ref));
		 const Point p1 = Point(w2l * (vertexPositions[idx[1]] - ref));
		 const Point p2 = Point(w2l * (vertexPositions[idx[2]] - ref));

		 Float sample1 = sqrt(sample.x);
	
		 Vector directionLocal = Vector(p0 * (1.0f - sample1) + p1 * sample1 * sample.y +
			 p2 * sample1 * (1.0f - sample.y));
		 
		 Vector directionWorld = l2w * directionLocal;
		 p = Point(directionWorld) + ref;
		 
		 // Note that we are not doing rejection sampling
		 // To do rejection sampling, if the triangle goes below horizon in local space, it must be clipped and the area/pdfNorm must be adjusted. Also we cannot have a sample below the horizon.
		 // Since we are not doing rejection sampling, if a direction is below the horizon in local space, it'll evaluate to zero when computing the brdf anyway, 
         // so we can return a zero pdf, even though the pdf is non-zero.
		 //if (directionLocal.z <= Epsilon)
			 //return 0.0f;
		 
		 directionWorld /= directionWorld.length();
		 
		 // compute pdf in local space
		 Float pdfLocal = 0;
		 {	
			 Float area = 0;
			 Normal nLocal = computeAreaNormal(area, p0, p1, p2);

			 Float dist = directionLocal.length();

			 directionLocal /= dist;

			 Float cosineFactor = -dot(directionLocal, nLocal);
			 if (cosineFactor <= Epsilon)
				 return 0.0f;

			 pdfLocal = dist * dist / (area * cosineFactor);
		 }

		 // compute the jacobian
		 Float jacobian = 1.0;
		 {
			 Vector w = w2l * directionWorld;
			 Float length = w.length();
			 jacobian = w2lDet / (length * length * length);
		 }
		 
		 return pdfLocal * jacobian;
	 }
 private:
	 // compute area and normal
	 Normal computeAreaNormal(Float &area, const Point &p0, const Point &p1, const Point &p2) const
	 {
		 Normal n = cross(p1 - p0, p2 - p0);
		 area = n.length();
		 n /= area;
		 area *= 0.5f;

		 return n;
	 }
 };
 
 struct EmitterSampler
 {
 public:
	 uint32_t triangleCount;
	 uint32_t vertexCount;
	 const Point *vertexPositions; // pointer to world positions
	 Spectrum radiance;

	 TriangleEmitters *triangleEmitters;

	 Float getDistanceFromCenter(const Point &ref, const Matrix3x3 &w2l = GET_MAT3x3_IDENTITY)
	 {	
		 Vector center(0.0f);
		 for (uint32_t i = 0; i < vertexCount; i++)
			 center += w2l * (vertexPositions[i] - ref);

		 center /= static_cast<Float>(vertexCount);

		 return center.length();
	 }

	 Float getSize(const Matrix3x3 &w2l = GET_MAT3x3_IDENTITY)
	 {
		 return getDistanceFromCenter(vertexPositions[0], w2l);
	 }

	 Float sample(Point &p, Normal &n, Sampler *sampler, const Point &ref,
		 const Matrix3x3 &w2l = GET_MAT3x3_IDENTITY,
		 Float w2lDet = 1.0f,
		 const Matrix3x3 &l2w = GET_MAT3x3_IDENTITY)
	 {
		 if (triangleCount != 2) {
			std::cerr << "Only supprts exactly two triangles" << std::endl;
			return 0.0f;
		 }
		
		 Float sample = sampler->next1D();
		
		 Float area0;
		 Float area1;

		 n = triangleEmitters[0].computeAreaNormal(area0, w2l);
		 triangleEmitters[1].computeAreaNormal(area1, w2l);
		
		 area0 /= (area0 + area1);
		 
		 if (sample <= area0)
			 return area0 * triangleEmitters[0].sample(p, sampler->next2D(), ref, w2l, w2lDet, l2w);
		 
		 return (1 - area0) * triangleEmitters[1].sample(p, sampler->next2D(), ref, w2l, w2lDet, l2w);
	 }
 };


 // Ideally one would adaptively sampling each source and run the filter in image space for each source seperately
 // One optimization would be combine similar size filters into one on per pixel basis.
 struct PerPixelData 
 {
	 Float *d1;
	 Float *d2Max; // Use for primal filter size 
	 Float *d2Min; // Use for computing adaptive sampling rates

	 Float *beta; // primal filter size
	 Float *sigma; // standard devation for each emitter, use size of emiiter / 2
	 Spectrum *colorEmitter; // color for each emiiter
	 Spectrum *colorEmitterBlur; // color for each emiiter after blurring
	 Spectrum color; // color for direct hit to light source
	 Point avgHitPoint; // avg primary ray hit point for each pixel
	 Vector avgShNormal; // Average shading normal at primary ray hit point

	 Float omegaMaxPix;
	 Float depth;
	 size_t *totalNumShadowSample;
	
	 void init(size_t nEmitters) 
	 {
		 d1 = new Float[nEmitters];
		 d2Max = new Float[nEmitters];
		 d2Min = new Float[nEmitters];

		 beta = new Float[nEmitters];
		 colorEmitter = new Spectrum[nEmitters];
		 colorEmitterBlur = new Spectrum[nEmitters];
		 totalNumShadowSample = new size_t[nEmitters];
		
		 for (size_t i = 0; i < nEmitters; i++) {
			 beta[i] = 0;
			 d1[i] = 0;
			 d2Max[i] = std::numeric_limits<Float>::min();
			 d2Min[i] = std::numeric_limits<Float>::max();
			 colorEmitter[i] = Spectrum(0.0f);
			 colorEmitterBlur[i] = Spectrum(0.0f);
			 totalNumShadowSample[i] = 0;
		 }
		 avgHitPoint = Point(0.0f);
		 depth = 0;
		 omegaMaxPix = 0.0f;
		 color = Spectrum(0.0f);
	 }
 };

class AAF2 : public Integrator {
public:
	AAF2(const Properties &props)
		: Integrator(props) {

      
        m_strictNormals = props.getBoolean("strictNormals", false);

        /**
         * When this flag is set to true, contributions from directly
         * visible emitters will not be included in the rendered image
         */
        m_hideEmitters = props.getBoolean("hideEmitters", false);
	}

	/// Unserialize from a binary data stream
	AAF2(Stream *stream, InstanceManager *manager)
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

	void collectEmitterParameters(Scene *scene)
	{
		auto emitters = scene->getEmitters();
		uint32_t numEmitters = 0;
		for (auto emitter : emitters)
		{
			if (!emitter->isOnSurface())
				std::cerr << "Ignoring light sources other than area light." << std::endl;
			else {
				if (emitter->getShape() == NULL)
					std::cerr << "Ignoring emitter with no shape." << std::endl;
				else if (typeid(*(emitter->getShape())) != typeid(TriMesh))
					std::cerr << "Ignoring emitter geometry other than TriMesh. RectMesh is possible but not yet supported." << std::endl;
				else
					numEmitters++;
			}
		}

		emitterCount = numEmitters;
		emitterSamplers = new EmitterSampler[numEmitters];

		numEmitters = 0;
		for (auto emitter : scene->getEmitters()) {
			if (emitter->isOnSurface() &&
				emitter->getShape() != NULL &&
				typeid(*(emitter->getShape())) == typeid(TriMesh)) {

				const TriMesh *triMesh = static_cast<const TriMesh *>(emitter->getShape());
				emitterSamplers[numEmitters].vertexCount = (uint32_t)triMesh->getVertexCount();
				emitterSamplers[numEmitters].triangleCount = (uint32_t)triMesh->getTriangleCount();
				emitterSamplers[numEmitters].vertexPositions = triMesh->getVertexPositions();

				emitterSamplers[numEmitters].triangleEmitters = new TriangleEmitters[triMesh->getTriangleCount()];
				const Triangle *triangles = triMesh->getTriangles();

				for (uint32_t i = 0; i < triMesh->getTriangleCount(); i++)
					emitterSamplers[numEmitters].triangleEmitters[i].init(triMesh->getVertexPositions(), triangles[i].idx[0], triangles[i].idx[1], triangles[i].idx[2]);
				
				emitterSamplers[numEmitters].radiance = emitter->getRadiance();
				numEmitters++;
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
		PerPixelData *ppd = new PerPixelData[cropSize.x * cropSize.y];

		collectEmitterParameters(scene);
	
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

			ppd[pixelID].init(emitterCount);
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
				collectSamples(rRec, gBuffer[pixelID *  sampleCount + k], ppd[pixelID]);
				sampler->advance();
			}

			for (size_t k = 0; k < emitterCount; k++) {
				ppd[pixelID].d1[k] /= (float)sampleCount;
			}

			ppd[pixelID].avgHitPoint /= (float)sampleCount;
			ppd[pixelID].avgShNormal /= (float)sampleCount;
			ppd[pixelID].color /= (float)sampleCount;
			ppd[pixelID].depth /= (float)sampleCount;
		});

		std::cout << "Finished Data-collection pass." << std::endl;

		runPool.run([&](int pixelID, int threadID) {
			ThreadData &td = threadData[threadID];
			auto sampler = td.sampler.get();
			RadianceQueryRecord rRec(scene, sampler);

			int i = pixelID % cropSize.x;
			int j = pixelID / cropSize.x;
			sampler->generate(Point2i(i, j));

			computeOmegaMaxPix(ppd, cropSize, pixelID);
			adaptiveSample(rRec, gBuffer[pixelID *  sampleCount], ppd[pixelID]);
			sampler->advance();
			computeBeta(rRec, gBuffer[pixelID *  sampleCount], ppd[pixelID]);
			screenSpaceBlur(rRec, cropSize, pixelID, ppd);
		});
		
		std::cout << "Finished adaptive sampling and blurring pass." << std::endl;

		pBufferToImage(result, ppd, cropSize, emitterCount);
		//gBufferToImage(result, gBuffer, cropSize, sampleCount);
        film->setBitmap(result);
		
		return true;
    }

	 void gBufferPass(const RayDifferential &primaryRay, RadianceQueryRecord &rRec, PrimaryRayData &prd) 
	 {
        Intersection &its = rRec.its;
		RayDifferential ray(primaryRay);
        bool intersect = rRec.rayIntersect(ray);
		ray.mint = Epsilon;
		
		prd.depth = 0;
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
		
		prd.depth = (its.p - primaryRay.o).length();
		prd.objectId = 0;
    }

	Float gaussian1D(Float x, Float sigma)
	{
		const float sqrt_2_pi = sqrt(2.f * M_PI);
		return exp(-(x * x) / (2.f * sigma * sigma)) / (sqrt_2_pi * sigma);
	}

	size_t nEmitterSamples = 4; //intial samples
	Float alpha = 1.0f; // bandlimit alpha
	Float mu = 2.0f;
	Float  maxAdaptiveSamples = 100.0f;
	Float gaussianSpreadCorrection = 3.0f;
	int maxFilterWidth = 10;

	Spectrum sampleDirect(uint32_t emitterIdx, DirectSamplingRecord &dRec, Sampler *sampler)
	{
		Float emitterPdf = emitterSamplers[emitterIdx].sample(dRec.p, dRec.n, sampler, dRec.ref);
		dRec.d = dRec.p - dRec.ref;
		dRec.dist = dRec.d.length();
		dRec.d /= dRec.dist;
		if (emitterPdf <= Epsilon || dot(dRec.d, dRec.refN) <= Epsilon)
			return Spectrum(0.0f);

		return emitterSamplers[emitterIdx].radiance / emitterPdf;
	}

	void collectSamples(RadianceQueryRecord &rRec, const PrimaryRayData &prd, PerPixelData &ppd) 
	{
		if (prd.objectId == -2)
			return;
		else if (prd.objectId == -1) {
			ppd.color += prd.its->Le(-prd.primaryRay->d);
			return;
		}
		
		ppd.avgHitPoint += prd.its->p;
		ppd.avgShNormal += prd.its->shFrame.n;
		ppd.depth += prd.depth;
				
		const Scene *scene = rRec.scene;
		DirectSamplingRecord dRec(*prd.its);
		Intersection its;
		
		for (uint32_t emitterIdx = 0; emitterIdx < emitterCount; emitterIdx++) {
				// compute the sum {I(y) V(y)} - here I(y) is gaussian light source
				Spectrum hitCount(0.0f);
				Float d1Avg = 0;
				Float d1Count = 0;
				for (size_t i = 0; i < nEmitterSamples; i++) {
					Spectrum value = sampleDirect(emitterIdx, dRec, rRec.sampler);
					RayDifferential shadowRay(prd.its->p, dRec.d, prd.primaryRay->time);
					shadowRay.mint = Epsilon;
					shadowRay.maxt = dRec.dist * (1 - ShadowEpsilon);
					bool intersectObject = scene->rayIntersect(shadowRay, its); // ray blocked by occluder
					
					BSDFSamplingRecord bRec(*prd.its, (*prd.its).toLocal(dRec.d));
					/* Evaluate BSDF * cos(theta) */
					const Spectrum bsdfVal = ((prd.its)->getBSDF(*prd.primaryRay))->eval(bRec);

					// apply gaussian falloff according to distance from center
					hitCount += intersectObject ? Spectrum(0.0f) : bsdfVal * value; //  gaussian1D(emitterSamplers[emitterIdx].getDistanceFromCenter(dRec.p), emitterSamplers[emitterIdx].getSize()) * emitterSamplers[emitterIdx].radiance; 
					
					// collect distance d1, d2Max, d2Min
					if (intersectObject && value.average() > 0) {
						d1Avg += abs(dot(dRec.n, (dRec.p - prd.its->p)));
						Float d2 = abs(dot(dRec.n, (dRec.p - its.p)));

						if (d2 < ppd.d2Min[emitterIdx])
							ppd.d2Min[emitterIdx] = d2;
						if (d2 > ppd.d2Max[emitterIdx])
							ppd.d2Max[emitterIdx] = d2;

						d1Count = d1Count + 1;
					}
				}
				
				ppd.d1[emitterIdx] = d1Count > 0 ? d1Avg / d1Count : 0;
				ppd.totalNumShadowSample[emitterIdx] += nEmitterSamples;
				ppd.colorEmitter[emitterIdx] += hitCount;
		}
	}

	void computeOmegaMaxPix(PerPixelData *ppd, const Vector2i &cropSize, int pixelID)
	{
		int i = pixelID % cropSize.x;
		int j = pixelID / cropSize.x;

		const Point &hitPoint = ppd[pixelID].avgHitPoint;
		Float d = 0.0f;
		if (Vector(hitPoint).length() > 0) {
			size_t pixLeft = pixelID - 1;
			size_t pixRight = pixelID + 1;
			size_t pixDn = (j - 1) * cropSize.x + i;
			size_t pixUp = (j + 1) * cropSize.x + i;
			Float ctr = 0;
			if (i > 0 && Vector(ppd[pixLeft].avgHitPoint).length() > 0) {
				d += Vector(ppd[pixLeft].avgHitPoint - hitPoint).length();
				ctr += 1;
			}
			if (i < (cropSize.x - 1) && Vector(ppd[pixRight].avgHitPoint).length() > 0) {
				d += Vector(ppd[pixRight].avgHitPoint - hitPoint).length();
				ctr += 1;
			}
			if (j > 0 && Vector(ppd[pixDn].avgHitPoint).length() > 0) {
				d += Vector(ppd[pixDn].avgHitPoint - hitPoint).length();
				ctr += 1;
			}
			if (j < (cropSize.y - 1) && Vector(ppd[pixUp].avgHitPoint).length() > 0) {
				d += Vector(ppd[pixUp].avgHitPoint - hitPoint).length();
				ctr += 1;
			}

			if (ctr > 0)
				d /= ctr;

			if (d > 0)
				ppd[pixelID].omegaMaxPix = 1.0f / d;
		}
	}

	void adaptiveSample(RadianceQueryRecord &rRec, const PrimaryRayData &prd, PerPixelData &ppd)
	{	
		if (prd.objectId == -2)
			return;
		else if (prd.objectId == -1)
			return;
		
		const Scene *scene = rRec.scene;
		DirectSamplingRecord dRec(*prd.its);
		dRec.ref = ppd.avgHitPoint;
		dRec.refN = ppd.avgShNormal;

		Intersection its;
		
		for (uint32_t emitterIdx = 0; emitterIdx < emitterCount; emitterIdx++) {
				// compute number of extra samples required i.e. adaptive sampling
				if (ppd.d2Max[emitterIdx] > 100 * std::numeric_limits<Float>::min()) {
					const Float s1 = std::max<Float>(ppd.d1[emitterIdx] / ppd.d2Min[emitterIdx], 1.f) - 1.f;
					Float s2 = std::max<Float>(ppd.d1[emitterIdx] / ppd.d2Max[emitterIdx], 1.f) - 1.f;
					Float inv_s2 = alpha / (1.f + s2);

					// Calculate pixel area and light area
					const float Ap = 1.f / (ppd.omegaMaxPix *ppd.omegaMaxPix);
					const float sigma = emitterSamplers[emitterIdx].getSize();
					const float Al = 4.f * sigma * sigma;

					// Calcuate number of additional samples
					const Float numSamples = std::min<Float>(4.f * pow(1.f + mu * (s1 / s2), 2.f) * pow(mu * 2 / s2 * sqrt(Ap / Al) + inv_s2, 2.f), maxAdaptiveSamples);
					size_t numAdaptiveSample = static_cast<size_t>(numSamples > 0 ? numSamples : 0);
					ppd.totalNumShadowSample[emitterIdx] += numAdaptiveSample;
					
					// compute the sum {I(y) V(y)} - here I(y) is gaussian light source
					Spectrum hitCount(0.0f);
					for (size_t i = 0; i < numAdaptiveSample; i++) {
						Spectrum value = sampleDirect(emitterIdx, dRec, rRec.sampler);
						RayDifferential shadowRay(ppd.avgHitPoint, dRec.d, prd.primaryRay->time);
						shadowRay.mint = Epsilon;
						shadowRay.maxt = dRec.dist * (1 - ShadowEpsilon);
						bool intersectObject = scene->rayIntersect(shadowRay, its); // ray blocked by occluder

						BSDFSamplingRecord bRec(*prd.its, (*prd.its).toLocal(dRec.d));
						// Evaluate BSDF * cos(theta) 
						const Spectrum bsdfVal = ((prd.its)->getBSDF(*prd.primaryRay))->eval(bRec);

						// apply gaussian falloff according to distance from center
						hitCount += intersectObject ? Spectrum(0.0f) : value * bsdfVal; //  gaussian1D(emitterSamplers[emitterIdx].getDistanceFromCenter(dRec.p), emitterSamplers[emitterIdx].getSize()) * emitterSamplers[emitterIdx].radiance;
					}

					ppd.colorEmitter[emitterIdx] += hitCount;
				}
		}
	
		// Normalize the colors
		for (size_t k = 0; k < emitterCount; k++)
			ppd.colorEmitter[k] /= static_cast<Float>(ppd.totalNumShadowSample[k]);
	}

	void computeBeta(RadianceQueryRecord &rRec, const PrimaryRayData &prd, PerPixelData &ppd)
	{
		if (prd.objectId == -2)
			return;
		else if (prd.objectId == -1)
			return;

		const Scene *scene = rRec.scene;

		int emitterIdx = 0;
		for (auto emitter : scene->getEmitters()) {
			if (emitter->isOnSurface() &&
				emitter->getShape() != NULL &&
				emitter->getShape()->getName().compare("rectangle") == 0) {

				if (ppd.d2Max[emitterIdx] > 100 * std::numeric_limits<Float>::min()) {
					// Update s2 and inv_s2
					const Float s2 = std::max<Float>(ppd.d1[emitterIdx] / ppd.d2Max[emitterIdx], 1.f) - 1.f;
					const Float inv_s2 = alpha / (1.f + s2);
					const Float omegaMaxX = inv_s2 * ppd.omegaMaxPix;

					const Float sigma = emitter->getShape()->getSize();

					// Calculate filter width at current pixel
					const float beta = (1.f / gaussianSpreadCorrection) * (1.f / mu) * std::max<Float>(sigma * s2, 1.f / omegaMaxX);
					ppd.beta[emitterIdx] = std::max<Float>(beta, std::numeric_limits<Float>::min());
				}

				emitterIdx++;
			}
		}
	}

	void screenSpaceBlur(RadianceQueryRecord &rRec, const Vector2i &cropSize, int pixelID, PerPixelData *ppd)
	{	
		const Scene *scene = rRec.scene;

		int i = pixelID % cropSize.x;
		int j = pixelID / cropSize.x;

		if (ppd[pixelID].depth < 100 * std::numeric_limits<Float>::min())
			return;

		int emitterIdx = 0;
		for (auto emitter : scene->getEmitters()) {
			if (emitter->isOnSurface() &&
				emitter->getShape() != NULL &&
				emitter->getShape()->getName().compare("rectangle") == 0)  {

				float beta = ppd[pixelID].beta[emitterIdx];

				if (beta > std::numeric_limits<Float>::min() && gaussian1D(0, beta) <= 1.0f) {
					//std::cout << beta << " " << gaussian1D(0, beta) << std::endl;
					Vector3 center = emitter->getShape()->getFrame().toLocal(Vector(ppd[pixelID].avgHitPoint));
					center.z = 0;
					Spectrum blurEmiiterColor(0.0f);
					Float weightNorm = 0.0f;
					for (int x = -maxFilterWidth; x < maxFilterWidth; x++) {
						int pixelX = i + x;
						if (pixelX < 0 || pixelX >= cropSize.x)
							continue;

						for (int y = -maxFilterWidth; y < maxFilterWidth; y++) {
							int pixelY = j + y;
							if (pixelY < 0 || pixelY >= cropSize.y)
								continue;

							int neighbourID = pixelY * cropSize.x + pixelX;

							if (abs(ppd[neighbourID].depth - ppd[pixelID].depth) / ppd[pixelID].depth > 0.1f)
								continue;

							Vector3 p = emitter->getShape()->getFrame().toLocal(Vector(ppd[neighbourID].avgHitPoint));
							p.z = 0;

							// Note that beta is not in pixel space, it is in the local coordinate of light source
							const Float w = gaussian1D((p - center).length(), beta);
							blurEmiiterColor += w * ppd[neighbourID].colorEmitter[emitterIdx];
							weightNorm += w;
						}
						if (weightNorm > 100 * std::numeric_limits<Float>::min())
							ppd[pixelID].colorEmitterBlur[emitterIdx] = blurEmiiterColor / weightNorm;
					}
				}
				else
					ppd[pixelID].colorEmitterBlur[emitterIdx] = ppd[pixelID].colorEmitter[emitterIdx];
			}
			emitterIdx++;
		}

	}

	/*
	Spectrum getUnshadowedIllumination(const Emitter *emitter, const Point &emitterHit, const Intersection &receiverIts, const BSDF *bsdf) 
	{
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
		
		return radiance * bsdf->eval(bRec) / r_2;
	}*/
   
	void gBufferToImage(ref<Bitmap> &result, const PrimaryRayData *gBuffer, const Vector2i &cropSize, const size_t sampleCount) 
	{
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

	void pBufferToImage(ref<Bitmap> &result, const PerPixelData *pBuffer, const Vector2i &cropSize, const uint32_t nEmitters) 
	{
		Spectrum *throughputPix = (Spectrum *)result->getData();
		
		for (size_t j = 0; j < cropSize.y; j++)
			for (size_t i = 0; i < cropSize.x; i++) {
				size_t currPix = j * cropSize.x + i;
				const PerPixelData &pData = pBuffer[currPix];
				throughputPix[currPix] = Spectrum(0.0f);
				throughputPix[currPix] = pData.color;

				// for unblurred results
				for (uint32_t k = 0; k < nEmitters; k++) {
					throughputPix[currPix] += pData.colorEmitter[k];
				}

				// For blurred results
				//for (uint32_t k = 0; k < nEmitters; k++) {
					//throughputPix[currPix] += pData.colorEmitterBlur[k];
				//}
				
				// Visualize d1
				// throughputPix[currPix] = Spectrum(pData.d1[0] / 200);

				// Visualize d2Max
				//throughputPix[currPix] = Spectrum(pData.d2Max[0]);

				// Visualize d2Min
				//if (pData.d2Min[0] < std::numeric_limits<Float>::max())
					//throughputPix[currPix] = Spectrum(pData.d2Min[0] / 200);

				// Visualize numAdaptiveSamples
				//for (uint32_t k = 0; k < emitterCount; k++)
					//throughputPix[currPix] += Spectrum(pData.totalNumShadowSample[k]);
				//throughputPix[currPix] /= nEmitters;
				//throughputPix[currPix] /= (nEmitterSamples + maxAdaptiveSamples);

				// visualize beta
				//for (uint32_t k = 0; k < nEmitters; k++)
					//throughputPix[currPix] += Spectrum(pData.beta[k]);
				//throughputPix[currPix] /= nEmitters;
				//throughputPix[currPix] /= (maxFilterWidth);

				// visualize depth
				//throughputPix[currPix] = Spectrum(pData.depth / 1000);

				// Visualize pixel footprint size
				//if (pData.omegaMaxPix > 0)
					//throughputPix[currPix] = Spectrum(1 / pData.omegaMaxPix);
				//else
					//throughputPix[currPix] = Spectrum(0.0f);
			}
	}
	
	void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "AAF2[" << endl
			<< "  strictNormals = " << m_strictNormals << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()

private:
    bool m_strictNormals;
    bool m_hideEmitters;
	
	EmitterSampler *emitterSamplers;
	uint32_t emitterCount;
};

MTS_IMPLEMENT_CLASS_S(AAF2, false, Integrator)
MTS_EXPORT_PLUGIN(AAF2, "AAF2");
MTS_NAMESPACE_END

// Issues found
// Since the we do not sample over the entire domain of gaussian falloff, the pdf is therefore unnormalized, hence looks brighter or darker depending on number of samples.
// Assumption that light source is gaussian is hard to justfy.
// Computing the form factor only at the center results in very gross approximation.
// Does not work in the areas of normal variation.
// Final gaussian blur is in the space w.r.t light source, hence it is difficult to estimate how many pixels needs to be searched. However, one can use some heuristics like when the weights fall below certain threshold we should stop.