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

#include <vector>
#include <functional>
#include <algorithm>
#include <thread>

#include <mitsuba/core/lock.h>
#include <mitsuba/core/thread.h>

#include "analytic.h"

MTS_NAMESPACE_BEGIN

#define GET_MAT3x3_IDENTITY ([](Matrix3x3 &m) {	m.setIdentity(); return static_cast<const Matrix3x3>(m); } (Matrix3x3(0.0f)))


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

struct EmitterTree;

struct BaseEmitter
{
	Point vertexPositions[3];
	Spectrum radiance;
};

// sample -> compuetEntropy -> partition -> evaluate
//   /\                            |
//    |                            | 
//    -----------------------------
struct EmitterNode 
{
	EmitterNode *nextNode[3] = {nullptr, nullptr, nullptr};
	uint32_t idx[3]; // index of the vertices of current node
	bool sampled = false; // set this to true when this node is sampled
	Float entropy; // compute this in computeEntropy, use neighbour pixels to compute
	uint32_t nBlack; // find this in computeEntropy, use only current pixel values;
	uint32_t nWhite; // find this in computeEntropy, use only current pixel values;
#define ENTROPY_THRESHOLD 0.001f
	
	void sample(const std::vector<Point2> &baryCoords, Sampler *sampler, std::vector<Point2> &samples, const BaseEmitter *emitter, uint32_t maxSamples)
	{	
		// recurse to leaf node
		uint32_t allNullPtr = (nextNode[0] == nullptr) + (nextNode[1] == nullptr) + (nextNode[2] == nullptr);
		if (allNullPtr == 0) {
			// recurse to leaf node
			nextNode[0]->sample(baryCoords, sampler, samples, emitter, maxSamples);
			nextNode[1]->sample(baryCoords, sampler, samples, emitter, maxSamples);
			nextNode[2]->sample(baryCoords, sampler, samples, emitter, maxSamples);

			return;
		}
		else if (allNullPtr < 3) {
			std::cerr << "Next node must be all nullptr or all must have some value" << std::endl;
			return;
		}

		// Do not sample if it is already sampled.
		if (sampled)
			return;

		const Point2 &a = baryCoords[idx[0]];
		const Point2 &b = baryCoords[idx[1]];
		const Point2 &c = baryCoords[idx[2]];
		
		for (uint32_t i = 0; i < maxSamples; i++) {
			Point2f rSample = sampler->next2D();
			Float sample1 = sqrt(rSample.x);

			samples.push_back(a * (1.0f - sample1) + b * sample1 * rSample.y +
				c * sample1 * (1.0f - rSample.y));
		}
				
		// compute the vertices of the adaptive-trinagle in world space
		/*const Point3 worldSpaceVertex0 = a.x * emitter->vertexPositions[0] + a.y * emitter->vertexPositions[1] + (1 - a.x - a.y) *  emitter->vertexPositions[2];
		const Point3 worldSpaceVertex1 = b.x * emitter->vertexPositions[0] + b.y * emitter->vertexPositions[1] + (1 - b.x - b.y) *  emitter->vertexPositions[2];
		const Point3 worldSpaceVertex2 = c.x * emitter->vertexPositions[0] + c.y * emitter->vertexPositions[1] + (1 - c.x - c.y) *  emitter->vertexPositions[2];

		for (uint32_t i = 0; i < maxSamples; i++) {
			Point2f rSample = sampler->next2D();
			//std::cout << rSample.x << " " << rSample.y << std::endl;
			Float sample1 = sqrt(rSample.x);

			Point3 p = worldSpaceVertex0 * (1.0f - sample1)  + worldSpaceVertex1 * sample1 * rSample.y +
				worldSpaceVertex2 * sample1 * (1.0f - rSample.y);

			Point2f bary;
			barycentric(p, emitter->vertexPositions[0], emitter->vertexPositions[1], emitter->vertexPositions[2], bary);
			samples.push_back(bary);
		}*/

		sampled = true;
	}

	void testSampling(const std::vector<Point2> &baryCoords, const std::vector<Point2> &samples)
	{
		uint32_t allNullPtr = (nextNode[0] == nullptr) + (nextNode[1] == nullptr) + (nextNode[2] == nullptr);
		if (allNullPtr == 0) {
			// recurse to leaf node
			nextNode[0]->testSampling(baryCoords, samples);
			nextNode[1]->testSampling(baryCoords, samples);
			nextNode[2]->testSampling(baryCoords, samples);
			return;
		}
		else if (allNullPtr < 3) {
			std::cerr << "Next node must be all nullptr or all must have some value" << std::endl;
			return;
		}

		const Point2 &a = baryCoords[idx[0]];
		const Point2 &b = baryCoords[idx[1]];
		const Point2 &c = baryCoords[idx[2]];

		std::vector<Point2> store;
		for (const auto &sample : samples) {
			if (pointInTriangle(sample, a, b, c)) {
				store.push_back(sample);
			}
		}

		if (store.size() > 2) {
			std::cout << "Problem sampling :" << a.x << " " << a.y << " " << b.x << " " << b.y << " " << c.x << " " << c.y << std ::endl;

			for (auto &s : store)
				std::cout << s.x << " " << s.y << std::endl;
		}
	}

	bool partition(std::vector<Point2> &baryCoords)
	{	
		uint32_t allNullPtr = (nextNode[0] == nullptr) + (nextNode[1] == nullptr) + (nextNode[2] == nullptr);
		if (allNullPtr == 0) {
			// recurse to leaf node
			uint32_t numPart = nextNode[0]->partition(baryCoords) + 
				nextNode[1]->partition(baryCoords) + 
				nextNode[2]->partition(baryCoords);

			return numPart > 0;
		}
		else if (allNullPtr < 3) {
			std::cerr << "Next node must be all nullptr or all must have some value" << std::endl;
			return false;
		}

		// Check if there is need for partion based on entropy
		if (entropy < ENTROPY_THRESHOLD)
			return false;

		// compute appropriate pivot location
		const Point2 &a = baryCoords[idx[0]];
		const Point2 &b = baryCoords[idx[1]];
		const Point2 &c = baryCoords[idx[2]];

		uint32_t indexPivot = static_cast<uint32_t>(baryCoords.size());
		// push the centroid as pivot for now
		baryCoords.push_back((a + b + c) / 3);
		
		nextNode[0] = new EmitterNode(indexPivot, idx[0], idx[1]);
		nextNode[1] = new EmitterNode(indexPivot, idx[1], idx[2]);
		nextNode[2] = new EmitterNode(indexPivot, idx[2], idx[0]);
		
		return true;
	}

	// recursively call analaytic
	Spectrum eval(const std::vector<Point2> &baryCoords,
		const BaseEmitter *emitter,
		const Point3 &receiverPos,
		const Matrix3x3 &rotMat,
		const Spectrum &diffuseComponent, const Spectrum &specularComponent,
		const Matrix3x3 &w2l_bsdf = GET_MAT3x3_IDENTITY,
		const float &amplitude = 1)
	{
		// Check if all nextNode are either null or have some value

		uint32_t allNullPtr = (nextNode[0] == nullptr) + (nextNode[1] == nullptr) + (nextNode[2] == nullptr);

		if (allNullPtr == 3) {
			if (nWhite == 0)
				return Spectrum(0.0f);

			return analytic(baryCoords, emitter, receiverPos, rotMat, diffuseComponent, specularComponent, w2l_bsdf, amplitude) * (static_cast<Float>(nWhite) / (nBlack + nWhite));
		}
		else if (allNullPtr > 0) {
			std::cerr << "Next node must be all nullptr or all must have some value" << std::endl;
			return Spectrum(0.0f);
		}
		else {
			return nextNode[0]->eval(baryCoords, emitter, receiverPos, rotMat, diffuseComponent, specularComponent, w2l_bsdf, amplitude) +
				nextNode[1]->eval(baryCoords, emitter, receiverPos, rotMat, diffuseComponent, specularComponent, w2l_bsdf, amplitude) +
				nextNode[2]->eval(baryCoords, emitter, receiverPos, rotMat, diffuseComponent, specularComponent, w2l_bsdf, amplitude);
		}

	}

	Spectrum analytic(const std::vector<Point2> &baryCoords, 
		const BaseEmitter *emitter, 
		const Point3 &receiverPos, 
		const Matrix3x3 &rotMat, 
		const Spectrum &diffuseComponent, const Spectrum &specularComponent,
		const Matrix3x3 &w2l_bsdf = GET_MAT3x3_IDENTITY, 
		const float &amplitude = 1)
	{	
		Spectrum sum(0.0f);

		// get the barycentric coords of the adaptive-triangle
		const Point2 &a = baryCoords[idx[0]];
		const Point2 &b = baryCoords[idx[1]];
		const Point2 &c = baryCoords[idx[2]];

		// compute the vertices of the adaptive-trinagle in world space
		Point3 worldSpaceVertex0 = a.x * emitter->vertexPositions[0] + a.y * emitter->vertexPositions[1] + (1 - a.x - a.y) *  emitter->vertexPositions[2];
		Point3 worldSpaceVertex1 = b.x * emitter->vertexPositions[0] + b.y * emitter->vertexPositions[1] + (1 - b.x - b.y) *  emitter->vertexPositions[2];
		Point3 worldSpaceVertex2 = c.x * emitter->vertexPositions[0] + c.y * emitter->vertexPositions[1] + (1 - c.x - c.y) *  emitter->vertexPositions[2];

		Vector3 localEdge0 = rotMat * (worldSpaceVertex0 - receiverPos);
		Vector3 localEdge1 = rotMat * (worldSpaceVertex1 - receiverPos);
		Vector3 localEdge2 = rotMat * (worldSpaceVertex2 - receiverPos);

		Float result = Analytic::integrate(localEdge0, localEdge1, localEdge2);

		// Note that each triangle is considered a light source, hence we apply single sided or double sided processing here.
		if (false) // One sided light source
			result = result > 0.0f ? result : 0.0f;
		else // double sided light source
			result = std::abs(result);

		sum = diffuseComponent * result * 0.5f * INV_PI;

		return sum * emitter->radiance;
	}
	void computeEntropy(const std::vector<Point2> &baryCoords, const BaseEmitter *emitter, const std::vector<Point2> &samples, const std::vector<bool> &visibility, float &entropyVis, const std::vector<const EmitterTree *> &neightbourSamples);

	EmitterNode(uint32_t i0, uint32_t i1, uint32_t i2)
	{
		idx[0] = i0;
		idx[1] = i1;
		idx[2] = i2;

		entropy = std::numeric_limits<Float>::max();
	}
private:
	static Float sign(const Point2f &p1, const Point2f &p2, const Point2f &p3)
	{
		return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
	}

	static bool pointInTriangle(const Point2f &pt, const Point2f &v1, const Point2f &v2, const Point2f &v3)
	{
		float d1, d2, d3;
		bool has_neg, has_pos;

		d1 = sign(pt, v1, v2);
		d2 = sign(pt, v2, v3);
		d3 = sign(pt, v3, v1);

		has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
		has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

		return !(has_neg && has_pos);
	}

	void barycentric(const Point &p, const Point &a, const Point &b, const Point &c, Point2f &bary)
	{
		Vector v0 = b - a, v1 = c - a, v2 = p - a;
		float d00 = dot(v0, v0);
		float d01 = dot(v0, v1);
		float d11 = dot(v1, v1);
		float d20 = dot(v2, v0);
		float d21 = dot(v2, v1);
		float denom = d00 * d11 - d01 * d01;
		bary.y = (d11 * d20 - d01 * d21) / denom;
		float t = (d00 * d21 - d01 * d20) / denom;
		bary.x = 1.0f - bary.y - t;
	}
#undef ENTROPY_THRESHOLD
};

struct EmitterTree
{	
	const BaseEmitter *baseEmitter = nullptr;
	std::vector<Point2f> baryCoords; // pivot points for partitioning
	// Next two are sample visibility pairs
	std::vector<Point2f> samples; // samples stored as barycentric coords
	std::vector<bool> visibility; // true == visible

	EmitterNode *root = nullptr;

	// This will recursively get 1 sample at each leaf node
	// put the bary-coord of sample in baryCoords
	// put the visibility in visibility 
	void sample(const Scene *scene, Sampler *sampler, const Normal &reciverNormal, const Point3 &reciverPos, uint32_t &samplesUsed, uint32_t maxSamples = 1)
	{	
		size_t oldIndex = samples.size();
		root->sample(baryCoords, sampler, samples, baseEmitter, maxSamples);
		size_t newIndex = samples.size();

		samplesUsed = static_cast<uint32_t>(newIndex - oldIndex);

		for (size_t i = oldIndex; i < newIndex; i++) {
			// raytrace and update visibility.
			// convert the light-samples from triangle-space to world space
			const Point2f &baryCord = samples[i];
			Point3 worldSpacePosition = baryCord.x * baseEmitter->vertexPositions[0] + 
				baryCord.y * baseEmitter->vertexPositions[1] + 
				(1 - baryCord.x - baryCord.y) *  baseEmitter->vertexPositions[2];

			Vector direction = worldSpacePosition - reciverPos;

			if (dot(reciverNormal, direction) < 0) {
				visibility.push_back(false);
				continue;
			}

			Float length = direction.length();
			direction /= length;

			RayDifferential shadowRay(reciverPos, direction, 0);
			shadowRay.mint = Epsilon;
			shadowRay.maxt = length * (1 - ShadowEpsilon);
			
			if (scene->rayIntersect(shadowRay))
				visibility.push_back(false);
			else
				visibility.push_back(true);
		}
	}

	void testSampling()
	{
		root->testSampling(baryCoords, samples);
	}

	void computeEntropy(float &entropy, const std::vector<const EmitterTree *> &neightbourSamples = std::vector<const EmitterTree *>())
	{
		root->computeEntropy(baryCoords, baseEmitter, samples, visibility, entropy, neightbourSamples);
	}

	// This will recursively go down to the leaf node and partition the leaf node if required.
	bool partition() 
	{
		return root->partition(baryCoords);
	}

	// This will recursively evaluate the tesselated triangles.
	Spectrum eval(const Point3 &receiverPos,
		const Matrix3x3 &rotMat,
		const Spectrum &diffuseComponent, const Spectrum &specularComponent,
		const Matrix3x3 &w2l_bsdf = GET_MAT3x3_IDENTITY,
		const float &amplitude = 1) 
	{
		return root->eval(baryCoords, baseEmitter, receiverPos, rotMat, diffuseComponent, specularComponent, w2l_bsdf, amplitude);
	}

	void init(const BaseEmitter *baseEmitterPtr)
	{
		baseEmitter = baseEmitterPtr;
		baryCoords.push_back(Point2(1, 0));
		baryCoords.push_back(Point2(0, 1));
		baryCoords.push_back(Point2(0, 0));
		root = new EmitterNode(0, 1, 2);
	}
};

void EmitterNode::computeEntropy(const std::vector<Point2> &baryCoords, const BaseEmitter *emitter, const std::vector<Point2> &samples, const std::vector<bool> &visibility, float &entropyVis, const std::vector<const EmitterTree *> &neightbourSamples)
{
	uint32_t allNullPtr = (nextNode[0] == nullptr) + (nextNode[1] == nullptr) + (nextNode[2] == nullptr);
	if (allNullPtr == 0) {
		// recurse to leaf node
		nextNode[0]->computeEntropy(baryCoords, emitter, samples, visibility, entropyVis, neightbourSamples);
		nextNode[1]->computeEntropy(baryCoords, emitter, samples, visibility, entropyVis, neightbourSamples);
		nextNode[2]->computeEntropy(baryCoords, emitter, samples, visibility, entropyVis, neightbourSamples);
		return;
	}
	else if (allNullPtr < 3) {
		std::cerr << "Next node must be all nullptr or all must have some value" << std::endl;
		return;
	}

	const Point2 &a = baryCoords[idx[0]];
	const Point2 &b = baryCoords[idx[1]];
	const Point2 &c = baryCoords[idx[2]];

	nWhite = 0;
	nBlack = 0;
	uint32_t idx = 0;
	for (const auto &sample : samples) {
		if (pointInTriangle(sample, a, b, c)) {
			nWhite += visibility[idx];
			nBlack += (!visibility[idx]);
		}
		idx++;
	}

	uint32_t nWhiteNeighbour = 0;
	uint32_t nBlackNeighbour = 0;
	for (const EmitterTree *root : neightbourSamples) {
		uint32_t idx = 0;
		for (const Point2f &sample : root->samples) {
			if (pointInTriangle(sample, a, b, c)) {
				nWhiteNeighbour += root->visibility[idx];
				nBlackNeighbour += (!(root->visibility[idx]));
			}
			idx++;
		}
	}

	if (nWhite + nWhiteNeighbour == 0)
		entropy = 0;
	else if (nBlack + nBlackNeighbour == 0)
		entropy = 0;
	else {
		float p = static_cast<Float>(nWhite + nWhiteNeighbour) / (nWhite + nBlack + nWhiteNeighbour + nBlackNeighbour);
		entropy = p * std::log(1.0f / p) + (1.0f - p) * std::log(1.0f / (1.0f - p));
	}

	entropyVis += entropy;// static_cast<Float>(nWhite) / (nWhite + nBlack);
}

/*
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

	 Spectrum getAnalytic(const Point &ref, const Matrix3x3 &rotMat, const Matrix3x3 &ltcW2l, const Float amplitude, const Spectrum &diffuseComponent, const Spectrum &specularComponent)
	 {	
		 Spectrum sum(0.0f);

		 Vector e0 = rotMat * (vertexPositions[idx[2]] - ref);
		 Vector e1 = rotMat * (vertexPositions[idx[1]] - ref);
		 Vector e2 = rotMat * (vertexPositions[idx[0]] - ref);

		 Float result = Analytic::integrate(e0, e1, e2);

		 // Note that each triangle is considered a light source, hence we apply single sided or double sided processing here.
		 if (true) // One sided light source
			 result = result > 0.0f ? result : 0.0f;
		 else // double sided light source
			 result = std::abs(result);

		 sum = diffuseComponent * result * 0.5f * INV_PI;

		 e0 = ltcW2l * e0;
		 e1 = ltcW2l * e1;
		 e2 = ltcW2l * e2;

		 result = Analytic::integrate(e0, e1, e2);
		
		 if (true) // One sided light source
			 result = result > 0.0f ? result : 0.0f;
		 else // double sided light source
			 result = std::abs(result);

		 sum += specularComponent * result * amplitude * 0.5f * INV_PI;

		 return sum;
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
 };
 */

struct PrimaryRayData 
{
	Intersection *its;
	RayDifferential *primaryRay;
	Float depth;
    int objectId;
};

struct PerPixelData
{
	EmitterTree *trees = nullptr;
	Spectrum colorDirect;
	Spectrum *colorShaded;
	uint32_t samplesUsed; // For visualization only
	float entropy; // For visualization only

	void init(const BaseEmitter *emitters, const uint32_t numEmitters)
	{
		trees = new EmitterTree[numEmitters];
		colorShaded = new Spectrum[numEmitters];

		for (uint32_t i = 0; i < numEmitters; i++) {
			trees[i].init(&emitters[i]);
			colorShaded[i] = Spectrum(0.0f);
		}
		samplesUsed = 0;
		entropy = 0;
		colorDirect = Spectrum(0.0f);
	}
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

		collectEmitterParameters(scene);

		// global buffer
		gBuffer = new PrimaryRayData[cropSize.x * cropSize.y];
		perPixelData = new PerPixelData[cropSize.x * cropSize.y];
					
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

			perPixelData[pixelID].init(emitters, emitterCount);
        });

		std::cout << "Finished GBuffer pass." << std::endl;

		runPool.run([&](int pixelID, int threadID) {
			ThreadData &td = threadData[threadID];
			auto sampler = td.sampler.get();
			RadianceQueryRecord rRec(scene, sampler);

			int i = pixelID % cropSize.x;
			int j = pixelID / cropSize.x;
			sampler->generate(Point2i(i, j));
			sample(rRec, gBuffer[pixelID], perPixelData[pixelID], 1);
			sampler->advance();
		});

		std::cout << "Finished initial sampling pass." << std::endl;

		runPool.run([&](int pixelID, int threadID) {
			int i = pixelID % cropSize.x;
			int j = pixelID / cropSize.x;
		
			std::vector<const EmitterTree *> tree;
			tree.reserve(8);
			computeEntropyCooperative(gBuffer[pixelID], perPixelData, i, j, cropSize, tree);
		});

		runPool.run([&](int pixelID, int threadID) {
			ThreadData &td = threadData[threadID];
			auto sampler = td.sampler.get();
			RadianceQueryRecord rRec(scene, sampler);

			int i = pixelID % cropSize.x;
			int j = pixelID / cropSize.x;
			sampler->generate(Point2i(i, j));
			sample(rRec, gBuffer[pixelID], perPixelData[pixelID], 1);
			sampler->advance();
			
		});

		runPool.run([&](int pixelID, int threadID) {
			int i = pixelID % cropSize.x;
			int j = pixelID / cropSize.x;

			std::vector<const EmitterTree *> tree;
			tree.reserve(8);
			computeEntropyCooperative(gBuffer[pixelID], perPixelData, i, j, cropSize, tree, 2);
		});

		runPool.run([&](int pixelID, int threadID) {
			ThreadData &td = threadData[threadID];
			auto sampler = td.sampler.get();
			RadianceQueryRecord rRec(scene, sampler);

			int i = pixelID % cropSize.x;
			int j = pixelID / cropSize.x;
			sampler->generate(Point2i(i, j));
			sample(rRec, gBuffer[pixelID], perPixelData[pixelID], 1);
			shadeAnalytic(gBuffer[pixelID], perPixelData[pixelID]);
			sampler->advance();

		});

		
		//gBufferToImage(result, gBuffer, cropSize);
		pBufferToImage(result, perPixelData, cropSize);
		film->setBitmap(result);
		
		return true;
    }

	void collectEmitterParameters(Scene *scene)
	{
		uint32_t numEmitters = 0;
		for (auto emitter : scene->getEmitters())
		{
			if (!emitter->isOnSurface())
				std::cerr << "Ignoring light sources other than area light." << std::endl;
			else {
				if (emitter->getShape() == NULL)
					std::cerr << "Ignoring emitter with no shape." << std::endl;
				else if (typeid(*(emitter->getShape())) != typeid(TriMesh))
					std::cerr << "Ignoring emitter geometry other than TriMesh. RectMesh is possible but not yet supported." << std::endl;
				else {
					const TriMesh *triMesh = static_cast<const TriMesh *>(emitter->getShape());
					numEmitters += (uint32_t)triMesh->getTriangleCount();
				}
			}
		}

		emitterCount = numEmitters;
		emitters = new BaseEmitter[numEmitters];

		numEmitters = 0;
		for (auto emitter : scene->getEmitters()) {
			if (emitter->isOnSurface() &&
				emitter->getShape() != NULL &&
				typeid(*(emitter->getShape())) == typeid(TriMesh)) {

				const TriMesh *triMesh = static_cast<const TriMesh *>(emitter->getShape());
				const Triangle *triangles = triMesh->getTriangles();
				const Point *vertexPositions = triMesh->getVertexPositions();
				for (uint32_t i = 0; i < triMesh->getTriangleCount(); i++) 
				{
					emitters[numEmitters].vertexPositions[0] = vertexPositions[triangles[i].idx[2]];
					emitters[numEmitters].vertexPositions[1] = vertexPositions[triangles[i].idx[1]];
					emitters[numEmitters].vertexPositions[2] = vertexPositions[triangles[i].idx[0]];
					emitters[numEmitters].radiance = emitter->getRadiance();
					numEmitters++;
				}
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

	void sample(RadianceQueryRecord &rRec, PrimaryRayData &prd, PerPixelData &ppd, uint32_t nSamples = 1)
	{
		if (prd.objectId < 0)
			return;

		const Scene *scene = rRec.scene;
		for (uint32_t i = 0; i < emitterCount; i++) {
			uint32_t samplesUsed = 0;
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, samplesUsed, nSamples);
		}
	}
	
	void computeEntropyCooperative(PrimaryRayData &prd, PerPixelData *ppd, int i, int j, const Vector2i &cropSize, std::vector<const EmitterTree *> &neighbours, uint32_t partionDepth = 1)
	{
		if (prd.objectId < 0)
			return; 
				
		for (uint32_t k = 0; k < emitterCount; k++) {
			neighbours.clear();

			if (i + 1 < cropSize.x) {
				int pixelID = j * cropSize.x + i + 1;
				neighbours.push_back(&ppd[pixelID].trees[k]);
			}
			if (i - 1 >= 0) {
				int pixelID = j * cropSize.x + i - 1;
				neighbours.push_back(&ppd[pixelID].trees[k]);
			}
			if (j + 1 < cropSize.y) {
				int pixelID = (j + 1) * cropSize.x + i;
				neighbours.push_back(&ppd[pixelID].trees[k]);

				if (i - 1 >= 0)
					neighbours.push_back(&ppd[pixelID - 1].trees[k]);
				if (i + 1 < cropSize.x)
					neighbours.push_back(&ppd[pixelID + 1].trees[k]);
			}
			if (j - 1 >= 0) {
				int pixelID = (j - 1) * cropSize.x + i;
				neighbours.push_back(&ppd[pixelID].trees[k]);
				
				if (i - 1 >= 0)
					neighbours.push_back(&ppd[pixelID - 1].trees[k]);
				if (i + 1 < cropSize.x)
					neighbours.push_back(&ppd[pixelID + 1].trees[k]);
			}

			float entropy = 0;
			int pixelID = j * cropSize.x + i;
			ppd[pixelID].trees[k].computeEntropy(entropy, neighbours);

			for (uint32_t p = 0; p < partionDepth; p++)
				ppd[pixelID].trees[k].partition();
			//ppd[pixelID].trees[k].partition();
			//ppd[pixelID].entropy += entropy;
		}
	}



	void shadeAnalytic(PrimaryRayData &prd, PerPixelData &ppd)
	{
		if (prd.objectId == -2)
			return;
		else if (prd.objectId == -1) {
			ppd.colorDirect += prd.its->Le(-prd.primaryRay->d);
			return;
		}
	
		Matrix3x3 ltcW2l = GET_MAT3x3_IDENTITY;
		Float amplitude = 1.0f;
		Float ltcW2lDet = 1.0f;
		Matrix3x3 rotMat = GET_MAT3x3_IDENTITY;
		Spectrum diffuseComponent(0.0f);
		Spectrum specularComponent(0.0f);

		if (!getMatrices(prd, rotMat, ltcW2l, ltcW2lDet, amplitude, diffuseComponent, specularComponent))
			return;

		for (uint32_t i = 0; i < emitterCount; i++) {
			float entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.colorShaded[i] = ppd.trees[i].eval(prd.its->p, rotMat, diffuseComponent, specularComponent);
			ppd.entropy += entropy;
		}
	}
	void shadeAnalyticTester(RadianceQueryRecord &rRec, PrimaryRayData &prd, PerPixelData &ppd)
	{
		if (prd.objectId == -2)
			return;
		else if (prd.objectId == -1) {
			ppd.colorDirect += prd.its->Le(-prd.primaryRay->d);
			return;
		}
		const Scene *scene = rRec.scene;
		
		/*for (uint32_t i = 0; i < emitterCount; i++) {
			uint32_t samplesUsed = 0;
			float entropy = 0;
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, samplesUsed, 9);
			ppd.trees[i].computeEntropy(entropy);
			ppd.trees[i].partition();
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, ppd.samplesUsed);
			entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.trees[i].partition();
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, ppd.samplesUsed);
			entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.trees[i].partition();
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, ppd.samplesUsed);
			entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.trees[i].partition();
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, ppd.samplesUsed);
			entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.trees[i].partition();
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, ppd.samplesUsed);
			entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.trees[i].partition();
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, ppd.samplesUsed);
			entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.trees[i].partition();
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, ppd.samplesUsed);
			entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.trees[i].partition();
			ppd.trees[i].sample(scene, rRec.sampler, prd.its->shFrame.n, prd.its->p, ppd.samplesUsed);
			entropy = 0;
			ppd.trees[i].computeEntropy(entropy);
			ppd.entropy += entropy;
			ppd.samplesUsed += samplesUsed;
		}*/
				
		Matrix3x3 ltcW2l = GET_MAT3x3_IDENTITY;
		Float amplitude = 1.0f;
		Float ltcW2lDet = 1.0f;
		Matrix3x3 rotMat = GET_MAT3x3_IDENTITY;
		Spectrum diffuseComponent(0.0f);
		Spectrum specularComponent(0.0f);

		if (!getMatrices(prd, rotMat, ltcW2l, ltcW2lDet, amplitude, diffuseComponent, specularComponent))
			return;

		for (uint32_t i = 0; i < emitterCount; i++) {
			ppd.colorShaded[i] = ppd.trees[i].eval(prd.its->p, rotMat, diffuseComponent, specularComponent);
		}
	}

	bool getMatrices(const PrimaryRayData &prd, Matrix3x3 &rotMat, Matrix3x3 &ltcW2l, Float &ltcW2lDet, Float &amplitude, Spectrum &diffuseComponent, Spectrum &specularComponent)
	{
		ltcW2l = GET_MAT3x3_IDENTITY;
		amplitude = 1.0f;
		ltcW2lDet = 1.0f;
		rotMat = GET_MAT3x3_IDENTITY;
		Float cosThetaIncident;
		Analytic::getRotMat(*prd.its, -prd.primaryRay->d, cosThetaIncident, rotMat);

		if (cosThetaIncident < 0)
			return false;

		Float thetaIncident = std::acos(cosThetaIncident);
		const BSDF *bsdf = ((prd.its)->getBSDF(*prd.primaryRay));
		diffuseComponent = bsdf->getDiffuseReflectance(*prd.its);
		specularComponent = Spectrum(0.0f);
		if (!bsdf->isDiffuse()) {
			bsdf->transform(*prd.its, thetaIncident, ltcW2l, amplitude);
			specularComponent = bsdf->getSpecularReflectance(*prd.its);
		}

		return true;
	}

	void pBufferToImage(ref<Bitmap> &result, const PerPixelData *pBuffer, const Vector2i &cropSize)
	{
		Spectrum *throughputPix = (Spectrum *)result->getData();

		for (size_t j = 0; j < cropSize.y; j++)
			for (size_t i = 0; i < cropSize.x; i++) {
				size_t currPix = j * cropSize.x + i;
				const PerPixelData &pData = pBuffer[currPix];
				throughputPix[currPix] = Spectrum(0.0f);
				throughputPix[currPix] = pData.colorDirect;

				// for unblurred results
				for (uint32_t k = 0; k < emitterCount; k++) {
					throughputPix[currPix] += pData.colorShaded[k];
				}
				//throughputPix[currPix] = Spectrum(pData.samplesUsed / (emitterCount * 20.0f));
				//throughputPix[currPix] = Spectrum(pData.entropy / (emitterCount * 50));
			}
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
	BaseEmitter *emitters = nullptr;
	PerPixelData *perPixelData = nullptr;
	PrimaryRayData *gBuffer = nullptr;
};

MTS_IMPLEMENT_CLASS_S(Entropy, false, Integrator)
MTS_EXPORT_PLUGIN(Entropy, "Entropy");
MTS_NAMESPACE_END