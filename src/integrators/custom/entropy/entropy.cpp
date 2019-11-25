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
 };

struct EmitterSampler
 {
 public:
	 uint32_t triangleCount;
	 uint32_t vertexCount;
	 const Point *vertexPositions; // pointer to world positions
	 Spectrum radiance;

	 TriangleEmitters *triangleEmitters;
 };

 struct PrimaryRayData 
 {
	Intersection *its;
	RayDifferential *primaryRay;
	Float depth;
    int objectId;
 };

 class Entropy : public Integrator {
public:
	Entropy(const Properties &props)
		: Integrator(props) {}

	/// Unserialize from a binary data stream
	Entropy(Stream *stream, InstanceManager *manager)
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
                int sceneResID, int sensorResID, int samplerResID) 
	{
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

		if (sampleCount != 1) {
			std::cout << "Sample count is not 1" << std::endl;
			return true;
		}

		// global buffer
		PrimaryRayData *gBuffer = new PrimaryRayData[cropSize.x * cropSize.y];
					
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

			rRec.newQuery(queryType, sensor->getMedium());
            Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

            if (needsApertureSample)
				apertureSample = rRec.nextSample2D();
            if (needsTimeSample)
                timeSample = rRec.nextSample1D();
			sensor->sampleRayDifferential(
				sensorRay, samplePos, apertureSample, timeSample);
			gBufferPass(sensorRay, rRec, gBuffer[pixelID]);
            sampler->advance();
        });

		std::cout << "Finished GBuffer pass." << std::endl;

		gBufferToImage(result, gBuffer, cropSize);
        film->setBitmap(result);
		
		return true;
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
   
	void gBufferToImage(ref<Bitmap> &result, const PrimaryRayData *gBuffer, const Vector2i &cropSize) 
	{
		int select = 1;

		Spectrum *throughputPix = (Spectrum *)result->getData();
		Spectrum value(0.0f);

		for (size_t j = 0; j < cropSize.y; j++)
			for (size_t i = 0; i < cropSize.x; i++) 
			{
				size_t currPix = j * cropSize.x + i;
				const PrimaryRayData &prd = gBuffer[currPix];
				if (prd.objectId < 0)
					continue;
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

	void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "Entropy[" << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
	uint32_t emitterCount = 0;
	EmitterSampler *emitterSamplers = nullptr;
};

MTS_IMPLEMENT_CLASS_S(Entropy, false, Integrator)
MTS_EXPORT_PLUGIN(Entropy, "Entropy");
MTS_NAMESPACE_END