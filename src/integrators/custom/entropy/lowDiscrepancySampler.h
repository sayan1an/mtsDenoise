#pragma once
#if !defined(__MITSUBA_PLUGIN_LOW_DISCREPANCY_H_)
#define __MITSUBA_PLUGIN_LOW_DISCREPANCY_H_
#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) > (b) ? (b) : (a))

class BaseSampler : public Object
{
public:
    virtual Point2f nextSquare() = 0;
    virtual Point2f nextTriangle() = 0;
    virtual Float next1D() = 0;
    virtual void seed(const Point2i &o) = 0;
};

class DefaultSampler : public BaseSampler
{
public:
    DefaultSampler(ref<Sampler> &s)
    {
        samplerRef = s;
        sampler = samplerRef.get(); 
    }

    void seed(const Point2i &o)
    {   
        sampler->generate(o);
    }

    Point2f nextSquare()
    {   
      return sampler->next2D();
    }

    Point2f nextTriangle()
    {   Point2f rSample = sampler->next2D();
		Float sample1 = sqrt(rSample.x);
		return Point2f(1.0f - sample1, sample1 * rSample.y);
	}

    Float next1D()
    {
        return sampler->next1D();
    }
private:
    ref<Sampler> samplerRef;
    Sampler *sampler;
};

class PoissionDiscSampler : public BaseSampler
{
public:
    PoissionDiscSampler(Float radius, uint32_t maxRetries)
    {   
        if (radius >= 1.0f)
            Log(EError, "%s", "PoissionDiscSampler : radius must be smaller than 1\n");

        this->radius = radius;
        this->maxRetries = maxRetries;

        cellSize = radius / std::sqrt(2.0f);
        nGridCells = static_cast<uint32_t>(1 / cellSize) + 1;

        grid = new Point2f[nGridCells * nGridCells];
        
        pointCache = new Point2f[nGridCells * nGridCells];
        pointCacheSize = 0;

        activeCache = new Point2f[nGridCells * nGridCells];
        activeCacheSize = 0;

        generator.seed(10);
		distribution = std::uniform_real_distribution<float>(0.0f, 1.0f);
    }
    
    void seed(const Point2i &o)
    {   
        generator.seed(o.x * 10000 + o.y);
    }

    Point2f nextSquare()
    {  
        if (pointCacheSize == 0)
            generateSamples();

        return pointCache[--pointCacheSize];
    }

    Point2f nextTriangle()
    {   
        if (pointCacheSize == 0)
            generateSamples();

        Point2f p = pointCache[--pointCacheSize];

        squareToTriangleEric(p);

        return p;
	}

    Float next1D()
    {
        return distribution(generator);
    }

private:
    uint32_t maxRetries;
    Float radius;

    Float cellSize;
    int nGridCells;

    Point2f *grid;

    Point2f *pointCache;
    uint32_t pointCacheSize;

    Point2f *activeCache;
    uint32_t activeCacheSize;

    std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution;
    
    inline void insertIntoGrid(const Point2f &p)
    { 
        uint32_t xindex = static_cast<uint32_t>(p.x / cellSize);
        uint32_t yindex = static_cast<uint32_t>(p.y / cellSize);
    
        grid[yindex * nGridCells + xindex] = p;
    }

    inline void addToPointCache(const Point2f &p)
    {
        pointCache[pointCacheSize] = p;
        pointCacheSize++;
    }

    inline void addToActiveCache(const Point2f &p)
    {
        activeCache[activeCacheSize] = p;
        activeCacheSize++;
    }

    inline void removeFromActiveCache(uint32_t index)
    {
        for (uint32_t i = index; i < activeCacheSize - 1; i++)
            activeCache[i] = activeCache[i + 1];
     
        activeCacheSize = activeCacheSize - 1;
    }

    inline bool isValidPoint(const Point2f &p)
    { 
        if (p.x < 0 || p.x >= 1 || p.y < 0 || p.y >= 1)
            return false;

        int xindex = static_cast<int>(p.x / cellSize);
        int yindex = static_cast<int>(p.y / cellSize);
        int i0 = MAX(yindex - 1, 0);
        int i1 = MIN(yindex + 1, nGridCells - 1);
        int j0 = MAX(xindex - 1, 0);
        int j1 = MIN(xindex + 1, nGridCells - 1);

        for (int i = i0; i < i1 + 1; i++)
            for (int j = j0; j < j1 + 1; j++) {
                int index = i*nGridCells + j;
                if (grid[index].x >=  0 && (grid[index] - p).length() < radius)
                    return false;
            }
  
        return true;
    }

    void generateSamples()
    {   
        for (int i = 0; i < nGridCells * nGridCells; i++)
            grid[i].x = -1.0f;

        Point2f point(distribution(generator), distribution(generator));
        insertIntoGrid(point);
        addToPointCache(point);
        addToActiveCache(point);

        while (activeCacheSize > 0)
        {
            uint32_t random_index = static_cast<uint32_t>(distribution(generator) * activeCacheSize);

            bool found = false;
            for (uint32_t tries = 0; tries < maxRetries; tries++) {
                Float theta = distribution(generator) * 2 * M_PI;
                Float new_radius = radius + distribution(generator) * radius;
                Point2f pnew(activeCache[random_index].x + new_radius * std::cos(theta),
                    activeCache[random_index].y + new_radius * std::sin(theta));
              
                if (!isValidPoint(pnew))
                    continue;
      
                insertIntoGrid(pnew);
                addToPointCache(pnew);
                addToActiveCache(pnew);
                found = true;
                break;
            }
            if (!found)
                removeFromActiveCache(random_index);
        }
    }

    inline void squareToTriangleEric(Point2f &p) const
    {
        if (p.y > p.x) {
            p.x *= 0.5f;
            p.y  -= p.x;
        }
        else {
            p.y *= 0.5f;
            p.x  -= p.y;
        }
    }


};

#undef MAX
#undef MIN
MTS_NAMESPACE_END
#endif