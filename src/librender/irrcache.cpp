#include <mitsuba/render/irrcache.h>

MTS_NAMESPACE_BEGIN

HemisphereSampler::HemisphereSampler(int M, int N) : m_M(M), m_N(N) {
	m_entries = new SampleEntry[m_M*m_N];
	m_uk = new Vector[m_N];
	m_vk = new Vector[m_N];
	m_vkMinus = new Vector[m_N];
}

HemisphereSampler::~HemisphereSampler() {
	delete[] m_entries;
	delete[] m_uk;
	delete[] m_vk;
	delete[] m_vkMinus;
}

void HemisphereSampler::generateDirections(const Intersection &its, Sampler *sampler) {
	for (unsigned int j=0; j<m_M; j++) {
		for (unsigned int k=0; k<m_N; k++) {
			SampleEntry &entry = m_entries[j*m_N + k];
			Point2 sample = sampler->independent2D();

			/* Sample uniformly wrt. projected solid angles */
			Float sinTheta2 = (j+sample.x)/m_M;
			Float cosTheta = std::sqrt(std::max((Float) 0, 1-sinTheta2));
			Float sinTheta = std::sqrt(sinTheta2);
			Float phi = 2*M_PI*(k+sample.y)/m_N;

			entry.d  = its.geoFrame.toWorld(Vector(sinTheta*std::cos(phi), 
				sinTheta*std::sin(phi), cosTheta));
			entry.cosTheta = cosTheta;
			entry.sinTheta = sinTheta;
			entry.dist = -1;
		}
	}

	/* Precompute planar vectors - see "Practical Global Illumination" by Jaroslav Krivanek 
	   and Pascal Gautron for more details on this notation */
	for (unsigned int k=0; k<m_N; k++) {
		Float phi     =  2*M_PI*(k+.5f)/m_N,
			  vk      =  phi - M_PI/2,
			  vkMinus = (2*M_PI*k)/m_N + M_PI/2;

		/* v_k plane vectors (centered) */
		m_vk[k] = its.geoFrame.toWorld(Vector(std::cos(vk), std::sin(vk), 0));

		/* v_{k-} (positioned at the start of the cell intervals) */
		m_vkMinus[k] = its.geoFrame.toWorld(Vector(std::cos(vkMinus), std::sin(vkMinus), 0));

		/* u_k plane vectors (centered) */
		m_uk[k] = its.geoFrame.toWorld(Vector(std::cos(phi), std::sin(phi), 0));
	}
}

void HemisphereSampler::process(const Intersection &its) {
	for (int i=0; i<3; ++i) {
		m_rGrad[i] = Spectrum(0.0f);
		m_tGrad[i] = Spectrum(0.0f);
	}
	m_E = Spectrum(0.0f);
	m_hMean = 0;
	m_hMin = std::numeric_limits<Float>::infinity();
	m_hMinRestricted = std::numeric_limits<Float>::infinity();

	Float invDists = 0;
	for (unsigned int j=0; j<m_M; j++) {
		const Float cosThetaMinus = std::sqrt(1-j/(Float)m_M),
					sinThetaMinus = std::sqrt(j/(Float)m_M),
					cosTheta      = std::sqrt(1-(j+.5f)/m_M),
					sinTheta      = std::sqrt((j+.5f)/m_M),
					cosThetaPlus  = std::sqrt(1-(j+1)/(Float)m_M),
					cosThetaDiff  = cosThetaMinus - cosThetaPlus,
					tanTheta      = sinTheta / cosTheta;
		for (unsigned int k=0; k<m_N; k++) {
			const SampleEntry &entry = m_entries[j*m_N + k];

			/* Rotational gradient - \pi/(MN) * \sum_{k=0}^{N-1}(v_k \sum_{j=0}^{M-1}) \tan\theta_j * L_{jk}) */
			for (int i=0; i<3; ++i) 
				m_rGrad[i] += entry.L * (-tanTheta * m_vk[k][i]);

			if (j>1) {
				/* Gradient in the u_k-direction */
				const SampleEntry &other = m_entries[(j-1)*m_N + k];
				const Float minDist = std::min(entry.dist, other.dist);
				if (minDist > 0) {
					const Float factor = (2*M_PI*cosThetaMinus*cosThetaMinus*sinThetaMinus)/(m_N*minDist);
					const Spectrum spec = (entry.L - other.L) * factor;
					for (int i=0; i<3; ++i)
						m_tGrad[i] += spec * m_uk[k][i];
				}
			}

			int kPrev = k-1;
			if (kPrev < 0)
				kPrev = m_N-1;

			/* Gradient in the v_k-direction */
			const SampleEntry &other = m_entries[j*m_N + kPrev];
			const Float minDist = std::min(entry.dist, other.dist);
			if (minDist > 0) {
				const Spectrum spec = (entry.L - other.L) * (cosTheta * cosThetaDiff
						/ (minDist * sinTheta));
				for (int i=0; i<3; ++i) 
					m_tGrad[i] += spec * m_vkMinus[k][i];
			}

			if (entry.dist > 0) {
				invDists += 1.0f / entry.dist;
				m_hMin = std::min(m_hMin, entry.dist);
				/* Discard rays close to the tangent plane */
				if (entry.cosTheta > 0.173f) // at least 10deg
					m_hMinRestricted = std::min(m_hMinRestricted, entry.dist);
			}
			m_E += entry.L;
		}
	}
	if (invDists > 0)
		m_hMean = (m_M*m_N) / invDists;
	for (int i=0; i<3; ++i) 
		m_rGrad[i] *= M_PI/(m_M*m_N);
	m_E *= M_PI / (m_M*m_N);
}

/* First pass of neighbor clamping */
struct clamp_self_functor {
	clamp_self_functor(const Point &p, Float &R0) : p(p), R0(R0) {
	}

	void operator()(IrradianceCache::Record *sample) {
		Float distance = (p - sample->p).length();
		R0 = std::min(R0, sample->originalR0 + distance);
	}

	Point p;
	Float &R0;
};

/* Second pass of neighbor clamping */
struct clamp_neighbors_functor {
	clamp_neighbors_functor(const Point &p, Float R0) : p(p), R0(R0) {
	}

	void operator()(IrradianceCache::Record *sample) {
		Float distance = (p - sample->p).length();
		Float distanceLimit = R0 + distance;

		if (sample->originalR0 > distanceLimit) {
			/* Update valid range and clamp back into the
			   permitted interval */
			sample->originalR0 = distanceLimit;
			sample->R0 = std::min(sample->R0_max, 
				std::max(sample->R0_min, sample->originalR0));
		}
	}

	Point p;
	Float R0;
};

/* Irradiance interpolation functor */
struct irr_interp_functor {
	irr_interp_functor(const Intersection &its, Float kappa, bool gradients) : its(its), 
		kappa(kappa), weightSum(0), gradients(gradients) {
	}

	void operator()(const IrradianceCache::Record *sample) {
		Float weight = sample->getWeight(its.p, its.geoFrame.n, kappa);

		if (weight == 0)
			return;

		Spectrum extrapolated = sample->E;
		if (gradients) {
			Vector crossN = cross(sample->n, its.geoFrame.n);
			Vector diff = its.p - sample->p;

			for (int i=0; i<SPECTRUM_SAMPLES; ++i) {
				for (int j=0; j<3; ++j) {
					extrapolated[i] += 
						crossN[j] * sample->rGrad[j][i]
						+ diff[j] * sample->tGrad[j][i];
				}
				extrapolated[i] = std::max(extrapolated[i], (Float) 0);
			}
		}
		E += weight * extrapolated;

		weightSum += weight;
	}

	const Intersection &its;
	Float kappa, weightSum;
	bool gradients;
	Spectrum E;
};

IrradianceCache::IrradianceCache(const AABB &aabb) 
 : m_octree(aabb) {
	/* Use the longest AABB axis as an estimate of the scene dimensions */
	m_sceneSize = (aabb.max-aabb.min)[aabb.getLargestAxis()];
	m_mutex = new Mutex();

	/* Reasonable default settings */
	setQuality(1.0f);
	useGradients(true);
	clampNeighbor(true);
	clampScreen(true);
	clampInfluence(0.005f, 64 * 0.005f);
}

IrradianceCache::IrradianceCache(Stream *stream, InstanceManager *manager) : 
	m_octree(AABB(stream)) {
	m_mutex = new Mutex();
	m_kappa = stream->readFloat();
	m_sceneSize = stream->readFloat();
	m_minDist = stream->readFloat();
	m_maxDist = stream->readFloat();
	m_clampScreen = stream->readBool();
	m_clampNeighbor = stream->readBool();
	m_useGradients = stream->readBool();
	unsigned int recordCount = stream->readUInt();
	m_records.reserve(recordCount);
	for (unsigned int i=0; i<recordCount; ++i) {
		Record *sample = new Record(stream);
		Float validRadius = sample->R0 / (2*m_kappa);
		m_octree.insert(sample, AABB(
			sample->p-Vector(1,1,1)*validRadius,
			sample->p+Vector(1,1,1)*validRadius
		));
		m_records.push_back(sample);
	}
}

IrradianceCache::~IrradianceCache() {
	for (size_t i=0; i<m_records.size(); ++i)
		delete m_records[i];
}

void IrradianceCache::serialize(Stream *stream, InstanceManager *manager) const {
	m_octree.getAABB().serialize(stream);
	stream->writeFloat(m_kappa);
	stream->writeFloat(m_sceneSize);
	stream->writeFloat(m_minDist);
	stream->writeFloat(m_maxDist);
	stream->writeBool(m_clampScreen);
	stream->writeBool(m_clampNeighbor);
	stream->writeBool(m_useGradients);
	stream->writeUInt((unsigned int) m_records.size());
	for (unsigned int i=0; i<m_records.size(); ++i)
		m_records[i]->serialize(stream);
}

IrradianceCache::Record *IrradianceCache::put(const RayDifferential &ray, const Intersection &its, 
		const HemisphereSampler &hs) {
	const Spectrum &E = hs.getIrradiance();
	TranslationalGradient tGrad;
	for (int i=0; i<3; ++i)
		tGrad[i] = hs.getTranslationalGradient()[i];
	Float R0 = hs.getMinimumDistanceRestricted();
	if (!E.isValid()) {
		Log(EWarn, "Invalid irradiance cache sample: %s", E.toString().c_str());
		return NULL;
	}
	Float R0_min = 0, R0_max = std::numeric_limits<Float>::infinity();

	/* Clamping suggested by Tabellion and Lamourlette ("An Approximate Global 
	   Illumination System for Computer Generated Films") */
	if (m_clampScreen && ray.hasDifferentials) {
		const Float d = -dot(its.geoFrame.n, its.p);
		const Float txRecip = dot(its.geoFrame.n, ray.rx.d),
		            tyRecip = dot(its.geoFrame.n, ray.ry.d);
		if (txRecip != 0 && tyRecip != 0) {
			// Ray distances traveled 
			const Float tx = -(dot(its.geoFrame.n, ray.rx.o) + d) / 
				txRecip;
			const Float ty = -(dot(its.geoFrame.n, ray.ry.o) + d) / 
				tyRecip;
			Point px = ray.rx(tx), py = ray.ry(ty);
			Float sqrtArea = std::sqrt(cross(px-its.p, py-its.p).length())*2;

			R0_min = 3.0f*sqrtArea;
			R0_max = 20.0f*sqrtArea;
		}
	}

	R0_min = std::max(R0_min, m_minDist);
	R0_max = std::min(R0_max, m_maxDist);

	if (m_useGradients) {
		/* Limit R0 by the gradient magnitude [Krivanek et al.] */
		for (int i=0; i<SPECTRUM_SAMPLES; ++i) {
			Vector grad(tGrad[0][i], tGrad[1][i], tGrad[2][i]);
			Float length = grad.length();
			if (length > Epsilon)
				R0 = std::min(R0, E[i]/grad.length());
		}

		/* Limit the translational gradient magnitude [Krivanek et al.] */
		for (int i=0; i<3; ++i)
			tGrad[i] = tGrad[i] * 
				std::min((Float) 1, hs.getMinimumDistance() / R0_min);
	}

	if (m_clampNeighbor) {
		/* Perform neighbor clamping [Krivanek et al.] to distribute 
		   geometric feature information amongst neighboring hss */
		clamp_self_functor clampSelf(its.p, R0);
		m_octree.searchSphere(BSphere(its.p, R0), clampSelf);
		clamp_neighbors_functor clampNeighbors(its.p, R0);
		m_octree.searchSphere(BSphere(its.p, R0), clampNeighbors);
	}

	Record *record = new Record();
	record->p = its.p;
	record->n = its.geoFrame.n;
	record->E = E;
	record->R0 = std::min(R0_max, std::max(R0_min, R0));
	record->originalR0 = R0;
	record->R0_min = R0_min;
	record->R0_max = R0_max;
	for (int i=0; i<3; ++i) {
		record->rGrad[i] = hs.getRotationalGradient()[i];
		record->tGrad[i] = tGrad[i];
	}
	insert(record);
	return record;
}

void IrradianceCache::insert(Record *record) {
	Float validRadius = record->R0 / (2*m_kappa);
	m_octree.insert(record, AABB(
		record->p-Vector(1,1,1)*validRadius,
		record->p+Vector(1,1,1)*validRadius
	));
	m_mutex->lock();
	m_records.push_back(record);
	m_mutex->unlock();
}

static StatsCounter irradHits("Irradiance cache", "Hits");
static StatsCounter irradMisses("Irradiance cache", "Misses");

bool IrradianceCache::get(const Intersection &its, Spectrum &E) const {
	irr_interp_functor functor(its, m_kappa, m_useGradients);
	m_octree.lookup(its.p, functor);

	if (functor.weightSum > 0) {
		E = functor.E / functor.weightSum;
		++irradHits;
		Assert(!E.isNaN());
		return true;
	}

	++irradMisses;
	return false;
}

void IrradianceCache::clampInfluence(Float min, Float max) {
	Assert(min > 0.0f && min < 1.0f && max > 0.0f
		&& max <= 1.0f && min < max);
	m_minDist = min * m_sceneSize;
	m_maxDist = max * m_sceneSize;
}

std::string IrradianceCache::toString() const {
	std::ostringstream oss;
	oss << "IrradianceCache[" << endl
		<< "  records = " << m_records.size() << "," << endl
		<< "  quality = " << m_kappa << "," << endl
		<< "  sceneSize = " << m_sceneSize << "," << endl
		<< "  minDist = " << m_minDist << "," << endl
		<< "  maxDist = " << m_maxDist << "," << endl
		<< "  clampScreen = " << m_clampScreen << "," << endl
		<< "  clampNeighbor = " << m_clampNeighbor << "," << endl
		<< "  useGradients = " << m_useGradients << endl
		<< "]";
	return oss.str();
}

MTS_IMPLEMENT_CLASS(HemisphereSampler, false, Object)
MTS_IMPLEMENT_CLASS_S(IrradianceCache, false, Object)
MTS_NAMESPACE_END