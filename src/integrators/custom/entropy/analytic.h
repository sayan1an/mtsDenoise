#pragma once
#if !defined(__MITSUBA_PLUGIN_ANALYTIC_H_)
#define __MITSUBA_PLUGIN_ANALYTIC_H_
#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

#define MAX(a, b) ((a) > (b) ? (a) : (b))

class Analytic : public Object {
public:
    // Assuming shading point at (0,0,0) with shading normal (0,0,1), and v1, v2 projected on the norm-ball.
    static Float integrateEdge(const Vector &v1, const Vector &v2) {
        Float cosTheta = math::clamp(dot(v1, v2), -1.0f, 1.0f);
        Float theta = std::acos(cosTheta);
        Float d = cross(v1,v2).z;
        Float t = theta/std::sin(theta);
        t = std::isnan(t) ? 1.0f : t;
        return d * t;
    }

    // integrate a clipped polygon
    static Float integrate(const Vector &a, const Vector &b, const Vector &c) {
        Vector quadProj[6]; // Vectrices of polygon projected on hemisphere
     
        int index = 0;
        
        // If all vertices on the light source are visible from shade point
        if (a.z >= 0 && b.z >= 0 && c.z >= 0) {
            Vector e0 = normalize(a);
            Vector e1 = normalize(b);
            Vector e2 = normalize(c);
            return integrateEdge(e0, e1) + integrateEdge(e1, e2) + integrateEdge(e2, e0);
        }
        // If none of the verices are visible from shade point
        else if (a.z <= 0 && b.z <=0 && c.z <= 0)
            return 0.0f;
        
        // Some but not all verices are visible from shade point.
        // It is very important to clip the light triangle before normalizing. 
        // i.e instead of clipping the triangle on surface of ball, clip it in full space and then project on ball. This is because when you project on unit himisphere, the distances are scaled non-linearly. 
        
        Vector intersection;
        if (a.z >= 0)
            quadProj[index++] = normalize(a);
        if (linePlaneIntersect(a, b, intersection))
            quadProj[index++] = normalize(intersection);
        if (b.z >= 0)
            quadProj[index++] = normalize(b);
        if (linePlaneIntersect(b, c, intersection))
            quadProj[index++] = normalize(intersection);
        if (c.z >= 0)
            quadProj[index++] = normalize(c);
        if (linePlaneIntersect(c, a, intersection))
            quadProj[index++] = normalize(intersection);
        
        Float result = 0.0f;
        for (int i = 0; i < index; i++)
            result += integrateEdge(quadProj[i], quadProj[(i+1)%index]);

        return result;
    }

    // integrate a clipped polygon and also get the area of the clipped polygon and normal to the polygon
    static Float integrateAndGetArea(const Vector &a, const Vector &b, const Vector &c, Float &area, Vector &nEmitter) {
        Vector quadProj[6]; // Vectrices of polygon projected on hemisphere
        Vector quad[6]; // Vertices of polygon before projecting on hemisphere, use for computing area of visible polygon.

        int index = 0;
        area = 0.0f;

        Vector _nEmitter = cross(a - c, c - b);
        Float nEmitterLength = _nEmitter.length();
        nEmitter = _nEmitter / nEmitterLength;

        // If all vertices on the light source are visible from shade point
        if (a.z >= 0 && b.z >= 0 && c.z >= 0) {
            area = 0.5f * nEmitterLength;
           
            Vector e0 = normalize(a);
            Vector e1 = normalize(b);
            Vector e2 = normalize(c);
            return integrateEdge(e0, e1) + integrateEdge(e1, e2) + integrateEdge(e2, e0);
        }
        // If none of the verices are visible from shade point
        else if (a.z <= 0 && b.z <=0 && c.z <= 0)
            return 0.0f;
        
        // Some but not all verices are visible from shade point.
        // It is very important to clip the light triangle before normalizing. 
        // i.e instead of clipping the triangle on surface of ball, clip it in full space and then project on ball. This is because when you project on unit himisphere, the distances are scaled non-linearly. 
        // See http://geomalgorithms.com/a01-_area.html, for computing area of polygon.
        Vector intersection;
        if (a.z >= 0) {
            quad[index] = a;
            quadProj[index++] = normalize(a);
        }
        if (linePlaneIntersect(a, b, intersection)) {
            quad[index] = intersection;
            quadProj[index++] = normalize(intersection);
        }
        if (b.z >= 0) {
            quad[index] = b;
            quadProj[index++] = normalize(b);
        }
        if (linePlaneIntersect(b, c, intersection)) {
            quad[index] = intersection;
            quadProj[index++] = normalize(intersection);
        }
        if (c.z >= 0) {
            quad[index] = c;
            quadProj[index++] = normalize(c);
        }
        if (linePlaneIntersect(c, a, intersection)) {
            quad[index] = intersection;
            quadProj[index++] = normalize(intersection);
        }
        
        Float result = 0.0f;
        Vector temp(0.0f);
        for (int i = 0; i < index; i++) {
            temp += cross(quad[i], quad[(i+1)%index]);
            result += integrateEdge(quadProj[i], quadProj[(i+1)%index]);
        }
        area = 0.5f * std::abs(dot(nEmitter, temp));
        
        return result;
    }
        
    // Find the intersection between line segment AB and plane with normal (0,0,1).
    // There are various possibilites: either A or B or both is on the palne, AB does not intersect internally.
    static bool linePlaneIntersect(const Vector &A,  const Vector &B, Vector &intersection) {
        Float eps = 1e-15;
       
        Float ABz = A.z * B.z;
        // either A or B or both is on the plane
        // or A and B both are on same side of plane.
        if (std::abs(ABz) <= eps || ABz > 0)
            return false;

        Float t = -A.z/(A.z - B.z);
        intersection.x = A.x + (A.x - B.x) * t;
        intersection.y = A.y + (A.y - B.y) * t;
        intersection.z = 0.0f;
        
        return true;
    }

     // w is expected to be unit length, in world coordinates
     // We want w in our new XY plane. i.e the plane containing normal and w is XY plane.
    static void getRotMat(const Intersection &its, const Vector &w, Float &cosTheta, Matrix3x3 &rotMat) {
        // First we need a coordinate system in which w has phi = 0(wrt new X axis) and normal is (0, 0, 1)
        cosTheta = dot(its.shFrame.n, w);
        if (cosTheta < 0) {
            rotMat.setZero();
            return;
        }

        // what happens when w == shFrame.n?
        // well in that case, the required specifications for wi are already met and we can use whatever local basis we have. 
        else if (cosTheta > 1 - 1e-10) {
            Matrix3x3 m(its.shFrame.s, its.shFrame.t, its.shFrame.n);
            m.transpose(rotMat);
            return;
        }

        Vector newX = normalize(w - its.shFrame.n * cosTheta); // row 0
        Vector newY = cross(its.shFrame.n, newX); // row 1
        //rotMat = Matrix3x3(newX, newY, its.shFrame.n);
        Matrix3x3 m(newX, newY, its.shFrame.n); // insert as columns
        m.transpose(rotMat); // make them rows
        //Log(EInfo, "%f and %f", rotMat.row(0).y, m.col(0).y);
    }

    // computes brdf * cosine
    static Spectrum approxBrdfEval(const BSDFSamplingRecord &bRec, const Matrix3x3 &mInv, const Float mInvDet, const Float &amplitude, const Spectrum &specular, const Spectrum diffuse) {
        if (Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0)
            return Spectrum(0.0f);
        
        Vector w_ = mInv * bRec.wo;
        Float length = w_.length();
        
        // Also note that if mInv is scaled by a scalar, it is still okay as length ^ 3 is proportional to mInv.det(). 
        Float jacobian = mInvDet / (length * length * length);

        return (specular * MAX(0, w_.z / length) * jacobian * amplitude + diffuse * Frame::cosTheta(bRec.wo)) * INV_PI;
    }

    static Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample, const Matrix3x3 &m, const Matrix3x3 &mInv, const Float mInvDet, const Float &amplitude, 
        const Spectrum &specular, const Float specularLum, const Spectrum &diffuse, const Float diffuseLum) {
        _pdf = 0;
        if (Frame::cosTheta(bRec.wi) <= 0)
            return Spectrum(0.0f);

        Point2 sample(_sample);
        Float probSpecular = specularLum / (specularLum + diffuseLum);
        
        bool choseSpecular = true;

        if (sample.y < probSpecular) {
                sample.y /= probSpecular;
        } else {
            sample.y = (sample.y - probSpecular) / (1 - probSpecular);
            choseSpecular = false;
        }
        
        bRec.wo = warp::squareToCosineHemisphere(sample);
        bRec.wo /= bRec.wo.length();

        bRec.sampledComponent = 1;
        bRec.sampledType = BSDF::EDiffuseReflection;

        Float pdfSpecular = 0, pdfDiffuse = 0;
        if (choseSpecular) {
            bRec.wo = m * bRec.wo; 

            bRec.wo /= bRec.wo.length();

            bRec.sampledComponent = 0;
            bRec.sampledType = BSDF::EGlossyReflection;
        }
        
        if (Frame::cosTheta(bRec.wo) <= 0 || std::isnan(bRec.wo.length()) || std::isinf(bRec.wo.length()) || std::abs(bRec.wo.length() - 1) > 1e-5) {
           return Spectrum(0.0f);
        }

        // compute pdfSpecular == D(w) * probSpecular
        Vector w_ = mInv * bRec.wo;
        Float length = w_.length();
        Float jacobian = mInvDet / (length * length * length);
        Float D = MAX(0, w_.z / length) * jacobian * INV_PI;
        pdfSpecular =  D * probSpecular;

        // compute pdfDiffuse
        pdfDiffuse = (1.0f - probSpecular) * warp::squareToCosineHemispherePdf(bRec.wo);
        
        _pdf = pdfDiffuse + pdfSpecular;
        
        // return bsdf(wo) * cos(wo) / pdf(wo);
        return _pdf <= Epsilon ? Spectrum(0.0f) : (specular * D * amplitude + diffuse * Frame::cosTheta(bRec.wo) * INV_PI) / _pdf;
    }

    static Float pdf(const BSDFSamplingRecord &bRec, const Matrix3x3 &mInv, const Float mInvDet, const Float specularLum, const Float diffuseLum) {
        if (Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0)
                return 0;
              
        Float probSpecular = specularLum / (specularLum + diffuseLum);

        Vector w_ = mInv * bRec.wo;
        Float length = w_.length();
        Float jacobian = mInvDet / (length * length * length);
        Float D = MAX(0, w_.z / length) * jacobian * INV_PI;
         
        return D * probSpecular + (1.0f - probSpecular) * warp::squareToCosineHemispherePdf(bRec.wo);
    }

#define ONE_SIDED_EMITTER true
    static Spectrum ltcIntegrate(const Scene *scene, const Point &shadePoint, const Matrix3x3 &rotMat, const Matrix3x3 &mInv, const Float mInvDet, const Float amplitude, const Spectrum &specularReflectance, const Spectrum &diffuseReflectance) {
        
        Spectrum Li(0.0f);

        for (auto emitter : scene->getEmitters()) {
            if (emitter->isOnSurface() && 
                emitter->getShape() != NULL && 
                typeid(*(emitter->getShape())) == typeid(TriMesh)) {
                
                const TriMesh *triMesh = static_cast<const TriMesh *>(emitter->getShape());
                const Triangle *triangles = triMesh->getTriangles();
                const Point *vertexPositions = triMesh->getVertexPositions();
        
                for (size_t i = 0; i < triMesh->getTriangleCount(); i++) {
                    Float resultDiffuse, resultSpecular;

                    // Vector between shade point to the vertices on the light source
                    // in specially designed local coordinate of shade point.
                    Vector e0 = (rotMat * (vertexPositions[triangles[i].idx[2]] - shadePoint));
                    Vector e1 = (rotMat * (vertexPositions[triangles[i].idx[1]] - shadePoint));
                    Vector e2 = (rotMat * (vertexPositions[triangles[i].idx[0]] - shadePoint));

                    resultDiffuse = Analytic::integrate(e0, e1, e2); 
                     // Note that each triangle is considered a light source, hence we apply single sided or double sided processing here.
                    if (ONE_SIDED_EMITTER) // One sided light source
                        resultDiffuse = resultDiffuse > 0.0f ? resultDiffuse : 0.0f;
                    else // double sided light source
                        resultDiffuse = std::abs(resultDiffuse);

                    // transform to bsdf space. Note that uniform scaling of mInv does not affect the integration result.
                    // Also clipping of polygons shouldn't be affected by uniform scaling as points above horizon always stays above horizon, no matter how far or close 
                    // they are from the shade point. In other words, solid angle formed by the polygon on hemisphere does not change with uniform scaling, hence result
                    // stays same.
                    e0 = mInv * e0;
                    e1 = mInv * e1;
                    e2 = mInv * e2;

                    resultSpecular = Analytic::integrate(e0, e1, e2);
                                        
                    // Note that each triangle is considered a light source, hence we apply single sided or double sided processing here.
                    if (ONE_SIDED_EMITTER) // One sided light source
                        resultSpecular = resultSpecular > 0.0f ? resultSpecular : 0.0f;
                    else // double sided light source
                        resultSpecular = std::abs(resultSpecular);
                   
                    Li += (specularReflectance * resultSpecular * amplitude + resultDiffuse * diffuseReflectance) * emitter->getRadiance();          
                }
            }
        }

        return Li * 0.5f * INV_PI;
    }

    static Spectrum ltcIntegrateAndSample(const Scene *scene, const Point &shadePoint, const Matrix3x3 &rotMat, const Matrix3x3 &mInv, const Float mInvDet, const Float amplitude, const Spectrum &specularReflectance, const Spectrum &diffuseReflectance,
        size_t nTriEmitters, Float *triEmitterAreaBuffer, Float *triEmitterAreaLumBuffer, Float &areaNormSpecular, Float &areaNormLumSpecular, Float &omegaSpecular, 
        Float &areaNormDiffuse, Float &areaNormLumDiffuse, Float &omegaDiffuse,
        Vector *triEmitterNormalBuffer, Vector *triEmitterVertexBuffer, Spectrum *triEmitterRadianceBuffer) {
        
        Spectrum Li(0.0f);
        size_t emitter_index = 0;
        areaNormDiffuse = 0; areaNormLumDiffuse = 0;
        areaNormSpecular = 0; areaNormLumSpecular = 0;
        for (auto emitter : scene->getEmitters()) {
            if (emitter->isOnSurface() && 
                emitter->getShape() != NULL && 
                typeid(*(emitter->getShape())) == typeid(TriMesh)) {
                
                const TriMesh *triMesh = static_cast<const TriMesh *>(emitter->getShape());
                const Triangle *triangles = triMesh->getTriangles();
                const Point *vertexPositions = triMesh->getVertexPositions();
        
                for (size_t i = 0; i < triMesh->getTriangleCount(); i++) {
                    Float resultDiffuse, resultSpecular;

                    triEmitterRadianceBuffer[emitter_index] = emitter->getRadiance();

                    // Vector between shade point to the vertices on the light source
                    // in specially designed local coordinate of shade point.
                    Vector e0 = (rotMat * (vertexPositions[triangles[i].idx[2]] - shadePoint));
                    Vector e1 = (rotMat * (vertexPositions[triangles[i].idx[1]] - shadePoint));
                    Vector e2 = (rotMat * (vertexPositions[triangles[i].idx[0]] - shadePoint));

                    triEmitterVertexBuffer[3 * emitter_index] = e0;
                    triEmitterVertexBuffer[3 * emitter_index + 1] = e1;
                    triEmitterVertexBuffer[3 * emitter_index + 2] = e2;

                    resultDiffuse = Analytic::integrateAndGetArea(e0, e1, e2, triEmitterAreaBuffer[emitter_index], triEmitterNormalBuffer[emitter_index]);
                    triEmitterAreaLumBuffer[emitter_index] = triEmitterAreaBuffer[emitter_index] * triEmitterRadianceBuffer[emitter_index].getLuminance();
                    areaNormDiffuse += triEmitterAreaBuffer[emitter_index];
                    areaNormLumDiffuse += triEmitterAreaLumBuffer[emitter_index];
                     // Note that each triangle is considered a light source, hence we apply single sided or double sided processing here.
                    if (ONE_SIDED_EMITTER) // One sided light source
                        resultDiffuse = resultDiffuse > 0.0f ? resultDiffuse : 0.0f;
                    else // double sided light source
                        resultDiffuse = std::abs(resultDiffuse);

                    // transform to bsdf space. Note that uniform scaling of mInv does not affect the integration result.
                    // Also clipping of polygons shouldn't be affected by uniform scaling as points above horizon always stays above horizon, no matter how far or close 
                    // they are from the shade point. In other words, solid angle formed by the polygon on hemisphere does not change with uniform scaling, hence result
                    // stays same.
                    e0 = mInv * e0;
                    e1 = mInv * e1;
                    e2 = mInv * e2;

                    triEmitterVertexBuffer[3 * nTriEmitters + 3 * emitter_index] = e0;
                    triEmitterVertexBuffer[3 * nTriEmitters + 3 * emitter_index + 1] = e1;
                    triEmitterVertexBuffer[3 * nTriEmitters + 3 * emitter_index + 2] = e2;

                    resultSpecular = Analytic::integrateAndGetArea(e0, e1, e2, triEmitterAreaBuffer[nTriEmitters + emitter_index], triEmitterNormalBuffer[nTriEmitters + emitter_index]);
                    triEmitterAreaLumBuffer[nTriEmitters + emitter_index] = triEmitterAreaBuffer[nTriEmitters + emitter_index] * triEmitterRadianceBuffer[emitter_index].getLuminance();
                    areaNormSpecular += triEmitterAreaBuffer[nTriEmitters + emitter_index];
                    areaNormLumSpecular += triEmitterAreaLumBuffer[nTriEmitters + emitter_index];

                    // Note that each triangle is considered a light source, hence we apply single sided or double sided processing here.
                    if (ONE_SIDED_EMITTER) // One sided light source
                        resultSpecular = resultSpecular > 0.0f ? resultSpecular : 0.0f;
                    else // double sided light source
                        resultSpecular = std::abs(resultSpecular);

                    Li += (specularReflectance * resultSpecular * amplitude + resultDiffuse * diffuseReflectance) * triEmitterRadianceBuffer[emitter_index];
                    emitter_index++;       
                }
            }
        }

        // collect some statistics for choosing a better heuristics for sampling.
        Vector centroidDiffuse(0.0f);
        Vector centroidSpecular(0.0f);
        for (size_t i = 0; i < emitter_index; i++) {
            centroidDiffuse += triEmitterVertexBuffer[3 * i];
            centroidDiffuse += triEmitterVertexBuffer[3 * i + 1];
            centroidDiffuse += triEmitterVertexBuffer[3 * i + 2];

            centroidSpecular += triEmitterVertexBuffer[3 * nTriEmitters + 3 * i];
            centroidSpecular += triEmitterVertexBuffer[3 * nTriEmitters + 3 * i + 1];
            centroidSpecular += triEmitterVertexBuffer[3 * nTriEmitters + 3 * i + 2];
        }

        centroidDiffuse /= (3 * (Float)nTriEmitters);
        centroidSpecular /= (3 * (Float)nTriEmitters);
        Float avgDistanceToSourceSpecular = centroidSpecular.length();
        Float avgDistanceToSourceDiffuse = centroidDiffuse.length();
        omegaSpecular = areaNormSpecular / (avgDistanceToSourceSpecular * avgDistanceToSourceSpecular);
        omegaDiffuse = areaNormDiffuse / (avgDistanceToSourceDiffuse * avgDistanceToSourceDiffuse);

        return Li * 0.5f * INV_PI;
    }
#define SPECULAR_CONTROL 0.3
    static void getEmitterSamples(const size_t nTriEmitters, const Float *triEmitterAreaBuffer, const Float *triEmitterAreaLumBuffer, const Float areaNormSpecular, const Float areaNormLumSpecular, const Float omegaSpecular,
        const Float areaNormDiffuse, const Float areaNormLumDiffuse, const Float omegaDiffuse, const Float cosThetaIncident, const Vector *triEmitterNormalBuffer, const Vector *triEmitterVertexBuffer, const Spectrum *triEmitterRadianceBuffer,
        const Matrix3x3 &m, const Matrix3x3 &mInv, const Float mInvDet, const Float &amplitude, 
        const Float specularLum, const Float diffuseLum,
        RadianceQueryRecord &rRec, 
        size_t nSamples, Spectrum *emitterSampleValues, Vector *emitterSampleDirections, Float *emitterPdf, size_t *emitterIndices) {
        
        Float probSpecularLum = specularLum  / (specularLum + diffuseLum); // reflectance heuristic
        Float probSpecularOmega = omegaSpecular / (omegaSpecular + omegaDiffuse); // area heuristic
        Float probSpecular = probSpecularLum * probSpecularOmega * (cosThetaIncident > SPECULAR_CONTROL ? 1 : std::pow(cosThetaIncident, 3)); 
        //SLog(EInfo, "%f %f %f", areaNormSpecularNormalized, areaNormDiffuse, mInvDet);
     
        for (size_t i = 0; i < nSamples; i++) {
            
            size_t offset = 0;
            Float areaNorm = areaNormDiffuse;
            Float areaNormLum = areaNormLumDiffuse;
            bool choseSpecular = false;
            Spectrum sampledRadiance;
            Vector sampledDirection;
            Float sampledPdf;

            emitterPdf[i] = 0;
            emitterSampleDirections[i] = Vector(0.0f);
            emitterSampleValues[i] = Spectrum(0.0f);
            emitterIndices[i] = -1;
            
            if (rRec.nextSample1D() < probSpecular) {
                offset = nTriEmitters;
                areaNorm = areaNormSpecular;
                areaNormLum = areaNormLumSpecular;
                choseSpecular = true;
            }

            if (areaNorm < Epsilon) {
                emitterSampleValues[i] = Spectrum(0.0f);
                emitterPdf[i] = 0;
                emitterSampleDirections[i] = Vector(0.0f);
                //SLog(EInfo, "%f %f", areaNormSpecular, areaNormDiffuse);
                continue;
            }
            
            sampleEmitter(nTriEmitters, offset, areaNorm, areaNormLum, triEmitterAreaBuffer, triEmitterAreaLumBuffer, triEmitterVertexBuffer, triEmitterNormalBuffer, triEmitterRadianceBuffer,
                rRec,
                sampledDirection, sampledRadiance, sampledPdf, emitterIndices[i]);
            
            if (sampledPdf < Epsilon) {
                emitterSampleValues[i] = Spectrum(0.0f);
                emitterPdf[i] = 0;
                emitterSampleDirections[i] = Vector(0.0f);
                //SLog(EInfo, "After sample: %f %f %f %i", probSpecular, areaNormSpecular, areaNormDiffuse, choseSpecular);
                continue;
            }
            
            Float pdfSpecular = 0;
            Float pdfDiffuse = 0;
            if (choseSpecular) {
                Vector directionLocal = m * sampledDirection; //brdf coord to local coord 
                directionLocal /= directionLocal.length();
                
                emitterSampleDirections[i] = directionLocal;

                Float tempLength = directionLocal.length();
                if (Frame::cosTheta(directionLocal) <= 0 || std::isnan(tempLength) || std::isinf(tempLength) || std::abs(tempLength - 1) > 1e-5) {
                    emitterSampleValues[i] = Spectrum(0.0f);
                    emitterPdf[i] = 0;
                    continue;
                }

                Vector w_ = mInv * directionLocal;
                Float length = w_.length();
                Float jacobian = mInvDet / (length * length * length);
                Float D =  sampledPdf * jacobian;
                pdfSpecular = D * probSpecular;
                pdfDiffuse = areaNormDiffuse > Epsilon ? (1 - probSpecular) * calcEmitterPdf(directionLocal, nTriEmitters, 0, areaNormDiffuse, areaNormLumDiffuse, triEmitterAreaBuffer, triEmitterAreaLumBuffer,
                     triEmitterNormalBuffer, triEmitterVertexBuffer)
                                : 0;
                // SLog(EInfo, "%f %f", pdfSpecular, pdfDiffuse);
            }

            else {
                if (areaNormSpecular > Epsilon) {
                    Vector w_ = mInv * sampledDirection;
                    Float length = w_.length();
                    Float jacobian = mInvDet / (length * length * length);
                    w_ /= length; 
                    Float D = calcEmitterPdf(w_, nTriEmitters, nTriEmitters, areaNormSpecular, areaNormLumSpecular, triEmitterAreaBuffer, triEmitterAreaLumBuffer, triEmitterNormalBuffer, triEmitterVertexBuffer) * jacobian;
                    pdfSpecular =  D * probSpecular;
                }
                emitterSampleDirections[i] = sampledDirection;
                pdfDiffuse = (1 - probSpecular) * sampledPdf;
            }
            
            emitterPdf[i] = pdfDiffuse + pdfSpecular;
            
            emitterSampleValues[i] = emitterPdf[i] <= Epsilon ? Spectrum(0.0f) : sampledRadiance / emitterPdf[i];
        }
    }

    // direction in local coordinates
    static Float emitterPdf(const Vector &directionLocal, const size_t nTriEmitters,
        const Float *triEmitterAreaBuffer, const Float *triEmitterAreaLumBuffer, const Float areaNormSpecular, const Float areaNormLumSpecular, const Float omegaSpecular, 
        const Float areaNormDiffuse, const Float areaNormLumDiffuse, const Float omegaDiffuse, const Float cosThetaIncident, const Vector *triEmitterNormalBuffer, const Vector *triEmitterVertexBuffer,
        const Matrix3x3 &mInv, const Float mInvDet, const Float specularLum, const Float diffuseLum) {

        Float probSpecularLum = specularLum  / (specularLum + diffuseLum);
        Float probSpecularOmega = omegaSpecular / (omegaSpecular + omegaDiffuse);
        Float probSpecular = probSpecularLum * probSpecularOmega * (cosThetaIncident > SPECULAR_CONTROL ? 1 : std::pow(cosThetaIncident, 3)); 

        Float pdfSpecular = 0, pdfDiffuse = 0;
        if (areaNormSpecular > Epsilon) {
            Vector w_ = mInv * directionLocal;
            Float length = w_.length();
            Float jacobian = mInvDet / (length * length * length);
            w_ /= length; 
            Float D = calcEmitterPdf(w_, nTriEmitters, nTriEmitters, areaNormSpecular, areaNormLumSpecular, triEmitterAreaBuffer, triEmitterAreaLumBuffer, triEmitterNormalBuffer, triEmitterVertexBuffer) * jacobian;
            pdfSpecular =  D * probSpecular;
        }
        
        if (areaNormDiffuse > Epsilon)
            pdfDiffuse = (1 - probSpecular) * calcEmitterPdf(directionLocal, nTriEmitters, 0, areaNormDiffuse, areaNormLumDiffuse, triEmitterAreaBuffer, triEmitterAreaLumBuffer, triEmitterNormalBuffer, triEmitterVertexBuffer);
    
        return pdfDiffuse + pdfSpecular;
    }

private:
    static inline Vector sampleTriangle(const Vector &e0, const Vector &e1, const Vector &e2, const Point2 &sample) {
        Float samplex = std::sqrt(sample.x);

        return e0 * (1.0f - samplex) + e1 * samplex * sample.y +
			e2 * samplex * (1.0f - sample.y);

    }
#define SAMPLE_LIGHT_AREA 1    // 0 - use surface area to select a light source, 1 - use surface area x radiance to select light source, 2 - radmoly pick a light source with uniform probability.
    // Note the direction must be in the same domain as the emitter vertices
    static Float calcEmitterPdf(const Vector &direction, const size_t nTriEmitters, const size_t offset, 
        const Float areaNormalization, const Float areaLumNormalization, const Float *triEmitterAreaBuffer, const Float *triEmitterAreaLumBuffer, const Vector *triEmitterNormalBuffer, const Vector *triEmitterVertexBuffer) {
        if (direction.z <= 0)
            return 0;
        
        Ray ray(Point(0.0), direction, 0);
     
        Float t = 0;
        bool notHit = true;
        size_t slectedEmitter = 0;
        Float cosineFactor = 0;
        for (size_t i = 0; i < nTriEmitters; i++) {
            Float u, v;
            // Try reinterpret_cast for vector to point conversion, should be faster!!
            if (Triangle::rayIntersect(Point(triEmitterVertexBuffer[3 * offset + 3 * i]), Point(triEmitterVertexBuffer[3 * offset + 3 * i + 1]), Point(triEmitterVertexBuffer[3 * offset + 3 * i + 2]), ray, u, v, t) && t > Epsilon) {
                cosineFactor = dot(direction, triEmitterNormalBuffer[offset + i]);
                if (!ONE_SIDED_EMITTER || (ONE_SIDED_EMITTER && cosineFactor < 0)) {
                    notHit = false;
                    slectedEmitter = i;
                    break;
                }
            }
        }

        if (notHit)
            return 0;
#if SAMPLE_LIGHT_AREA == 0      
        Float omega = areaNormalization * std::abs(cosineFactor) / (t * t);
#elif SAMPLE_LIGHT_AREA == 1
        Float omega = triEmitterAreaBuffer[offset + slectedEmitter] * areaLumNormalization * std::abs(cosineFactor) / (t * t * triEmitterAreaLumBuffer[offset + slectedEmitter]);
#else
        Float omega = triEmitterAreaBuffer[offset + slectedEmitter] * nTriEmitters * std::abs(cosineFactor) / (t * t);
#endif  
         
        return omega > Epsilon ? 1 / omega : 1 / Epsilon;
    }

    static void sampleEmitter(const size_t nTriEmitters, const size_t offset, const Float areaNormalization, const Float areaLumNormalization, const Float *triEmitterAreaBuffer, const Float *triEmitterAreaLumBuffer, const Vector *triEmitterVertexBuffer, const Vector *triEmitterNormalBuffer, const Spectrum *triEmitterRadianceBuffer,
        RadianceQueryRecord &rRec, 
        Vector &sampledDirection, Spectrum &sampledValue, Float &sampledPdf, size_t &emitterIndex) {
        
        int maxTries = 10;
        // Select an emitter
        size_t slectedEmitter = 0;
        do {
            Float sample = rRec.nextSample1D();
            
            Float cdf = 0;
            for (size_t j = 0; j < nTriEmitters; j++) {
                Float _cdf = cdf;
#if SAMPLE_LIGHT_AREA == 0
                cdf += triEmitterAreaBuffer[offset + j] / areaNormalization;
#elif SAMPLE_LIGHT_AREA == 1
                cdf += triEmitterAreaLumBuffer[offset + j] / areaLumNormalization;
#else                
                cdf += 1 / (Float)nTriEmitters;
#endif
                if (_cdf <= sample && sample <= cdf) {
                    slectedEmitter = j;
                    break;
                }
            }
            sampledDirection = sampleTriangle(triEmitterVertexBuffer[3 * offset + 3 * slectedEmitter],
                triEmitterVertexBuffer[3 * offset + 3 * slectedEmitter + 1], 
                triEmitterVertexBuffer[3 * offset + 3 * slectedEmitter + 2], rRec.nextSample2D());
        
        } while(sampledDirection.z <= 0 && --maxTries > 0);

        Float length = sampledDirection.length();
        sampledDirection /= length;
        sampledValue = triEmitterRadianceBuffer[slectedEmitter];
        Float cosineFactor = dot(sampledDirection, triEmitterNormalBuffer[offset + slectedEmitter]);

        if (sampledDirection.z <= 0 || (ONE_SIDED_EMITTER && cosineFactor >= 0)) {
            sampledPdf = 0;
            sampledValue = Spectrum(0.0f);
            emitterIndex = -1;
        }
        else {
#if SAMPLE_LIGHT_AREA == 0            
            Float omega = areaNormalization * std::abs(cosineFactor) / (length * length);
#elif SAMPLE_LIGHT_AREA == 1
            Float omega = triEmitterAreaBuffer[offset + slectedEmitter] * areaLumNormalization * std::abs(cosineFactor) / (length * length * triEmitterAreaLumBuffer[offset + slectedEmitter]);
#else
            Float omega = triEmitterAreaBuffer[offset + slectedEmitter] * nTriEmitters * std::abs(cosineFactor) / (length * length);
#endif      
           
            sampledPdf =  omega > Epsilon ? 1 / omega : 1 / Epsilon;
            emitterIndex = slectedEmitter;
        }
    }
#undef SAMPLE_LIGHT_AREA    
#undef ONE_SIDED_EMITTER
};
MTS_NAMESPACE_END
#endif