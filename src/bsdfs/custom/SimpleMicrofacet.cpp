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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>
#include "../ior.h"
#include "../microfacet.h"

MTS_NAMESPACE_BEGIN

class SimpleMicrofacet : public BSDF {
public:
    SimpleMicrofacet(const Properties &props) : BSDF(props) {
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));
        m_diffuseReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("diffuseReflectance", Spectrum(0.5f)));

        /* Specifies the internal index of refraction at the interface */
        Float intIOR = lookupIOR(props, "intIOR", "benzene");

        /* Specifies the external index of refraction at the interface */
        Float extIOR = lookupIOR(props, "extIOR", "air");

        if (intIOR < 0 || extIOR < 0 || intIOR == extIOR)
            Log(EError, "The interior and exterior indices of "
                "refraction must be positive and differ!");

        m_eta = intIOR / extIOR;

        MicrofacetDistribution distr(props);
        m_type = distr.getType();
        m_sampleVisible = distr.getSampleVisible();

        if (distr.isAnisotropic())
            Log(EError, "The 'roughplastic' plugin currently does not support "
                "anisotropic microfacet distributions!");

        m_alpha = new ConstantFloatTexture(distr.getAlpha());

        m_specularSamplingWeight = 0.0f;
    }

    SimpleMicrofacet(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_type = (MicrofacetDistribution::EType) stream->readUInt();
        m_sampleVisible = stream->readBool();
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_diffuseReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_alpha = static_cast<Texture *>(manager->getInstance(stream));
        m_eta = stream->readFloat();

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        stream->writeUInt((uint32_t) m_type);
        stream->writeBool(m_sampleVisible);
        manager->serialize(stream, m_specularReflectance.get());
        manager->serialize(stream, m_diffuseReflectance.get());
        manager->serialize(stream, m_alpha.get());
        stream->writeFloat(m_eta);
    }

    void configure() {
        bool constAlpha = m_alpha->isConstant();

        m_components.clear();

        m_components.push_back(EGlossyReflection | EFrontSide
            | ((constAlpha && m_specularReflectance->isConstant())
                ? 0 : ESpatiallyVarying));
        m_components.push_back(EDiffuseReflection | EFrontSide
            | ((constAlpha && m_diffuseReflectance->isConstant())
                ? 0 : ESpatiallyVarying));

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);
        m_diffuseReflectance = ensureEnergyConservation(
            m_diffuseReflectance, "diffuseReflectance", 1.0f);
   
        /* Compute weights that further steer samples towards
           the specular or diffuse components */
        Float dAvg = m_diffuseReflectance->getAverage().getLuminance(),
              sAvg = m_specularReflectance->getAverage().getLuminance();
        m_specularSamplingWeight = sAvg / (dAvg + sAvg);

        m_usesRayDifferentials =
            m_specularReflectance->usesRayDifferentials() ||
            m_diffuseReflectance->usesRayDifferentials() ||
            m_alpha->usesRayDifferentials();

        BSDF::configure();
    }

    Spectrum getDiffuseReflectance(const Intersection &its) const {
        return m_diffuseReflectance->eval(its);
    }

    Spectrum getSpecularReflectance(const Intersection &its) const {
        return m_specularReflectance->eval(its);
    }

    /// Helper function: reflect \c wi with respect to a given surface normal
    inline Vector reflect(const Vector &wi, const Normal &m) const {
        return 2 * dot(wi, m) * Vector(m) - wi;
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
            (bRec.component == -1 || bRec.component == 0);
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
            (bRec.component == -1 || bRec.component == 1);

        if (measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 ||
            (!hasSpecular && !hasDiffuse))
            return Spectrum(0.0f);

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            m_type,
            m_alpha->eval(bRec.its).average(),
            m_sampleVisible
        );

        Spectrum result(0.0f);
        if (hasSpecular) {
            /* Calculate the reflection half-vector */
            const Vector H = normalize(bRec.wo+bRec.wi);

            /* Evaluate the microfacet normal distribution */
            const Float D = distr.eval(H);

            /* Fresnel term */
            /* set it to 1 when comparing with LTC */
            const Float F = 1;//fresnelDielectricExt(dot(bRec.wi, H), m_eta);

            /* Smith's shadow-masking function */
            const Float G = distr.G(bRec.wi, bRec.wo, H);

            /* Calculate the specular reflection component */
            Float value = F * D * G /
                (4.0f * Frame::cosTheta(bRec.wi));

            //if ( m_specularReflectance->getAverage().getLuminance() > 0.5f)
           // std::cout <<  m_specularReflectance->getAverage().getLuminance() << " " << m_diffuseReflectance->getAverage().getLuminance() << " " << hasDiffuse << " " << hasSpecular << std::endl;
            result += m_specularReflectance->eval(bRec.its) * value;
        }

        if (hasDiffuse)
            result += m_diffuseReflectance->eval(bRec.its) * (INV_PI * Frame::cosTheta(bRec.wo));
        

        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
            (bRec.component == -1 || bRec.component == 0);
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
            (bRec.component == -1 || bRec.component == 1);

        if (measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 ||
            (!hasSpecular && !hasDiffuse))
            return 0.0f;

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            m_type,
            m_alpha->eval(bRec.its).average(),
            m_sampleVisible
        );

        /* Calculate the reflection half-vector */
        const Vector H = normalize(bRec.wo+bRec.wi);

        Float probDiffuse, probSpecular;
        if (hasSpecular && hasDiffuse) {
            probSpecular = m_specularSamplingWeight;
            probDiffuse = 1 - probSpecular;
        } else {
            probDiffuse = probSpecular = 1.0f;
        }

        Float result = 0.0f;
        if (hasSpecular) {
            /* Jacobian of the half-direction mapping */
            const Float dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, H));

            /* Evaluate the microfacet model sampling density function */
            const Float prob = distr.pdf(bRec.wi, H);

            result = prob * dwh_dwo * probSpecular;
        }

        if (hasDiffuse)
            result += probDiffuse * warp::squareToCosineHemispherePdf(bRec.wo);

        return result;
    }

    inline Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample) const {
        bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
            (bRec.component == -1 || bRec.component == 0);
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
            (bRec.component == -1 || bRec.component == 1);

        if (Frame::cosTheta(bRec.wi) <= 0 || (!hasSpecular && !hasDiffuse))
            return Spectrum(0.0f);

        bool choseSpecular = hasSpecular;
        Point2 sample(_sample);

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            m_type,
            m_alpha->eval(bRec.its).average(),
            m_sampleVisible
        );

        Float probSpecular;
        if (hasSpecular && hasDiffuse) {
            probSpecular = m_specularSamplingWeight;
            if (sample.y < probSpecular) {
                sample.y /= probSpecular;
            } else {
                sample.y = (sample.y - probSpecular) / (1 - probSpecular);
                choseSpecular = false;
            }
        }

        if (choseSpecular) {
            /* Perfect specular reflection based on the microfacet normal */
            Normal m = distr.sample(bRec.wi, sample);
            bRec.wo = reflect(bRec.wi, m);
            bRec.sampledComponent = 0;
            bRec.sampledType = EGlossyReflection;

            /* Side check */
            if (Frame::cosTheta(bRec.wo) <= 0)
                return Spectrum(0.0f);
        } else {
            bRec.sampledComponent = 1;
            bRec.sampledType = EDiffuseReflection;
            bRec.wo = warp::squareToCosineHemisphere(sample);
        }
        bRec.eta = 1.0f;

        /* Guard against numerical imprecisions */
        _pdf = pdf(bRec, ESolidAngle);

        if (_pdf == 0)
            return Spectrum(0.0f);
        else
            return eval(bRec, ESolidAngle) / _pdf;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        Float pdf;
        return SimpleMicrofacet::sample(bRec, pdf, sample);
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        BSDF::addChild(name, child);
    }

    Float getRoughness(const Intersection &its, int component) const {
        Assert(component == 0 || component == 1);

        if (component == 0)
            return m_alpha->eval(its).average();
        else
            return std::numeric_limits<Float>::infinity();
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "SimpleMicrofacet[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
            << "  sampleVisible = " << m_sampleVisible << "," << endl
            << "  alpha = " << indent(m_alpha->toString()) << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  diffuseReflectance = " << indent(m_diffuseReflectance->toString()) << "," << endl
            << "  specularSamplingWeight = " << m_specularSamplingWeight << "," << endl
            << "  diffuseSamplingWeight = " << (1-m_specularSamplingWeight) << "," << endl
            << "  eta = " << m_eta << "," << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    MicrofacetDistribution::EType m_type;
    ref<Texture> m_diffuseReflectance;
    ref<Texture> m_specularReflectance;
    ref<Texture> m_alpha;
    Float m_specularSamplingWeight;
    Float m_eta;
    bool m_sampleVisible;
};

MTS_IMPLEMENT_CLASS_S(SimpleMicrofacet, false, BSDF)
MTS_EXPORT_PLUGIN(SimpleMicrofacet, "Simple microfacet BRDF");
MTS_NAMESPACE_END
