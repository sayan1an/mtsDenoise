#if !defined(__MAXEXP_H)
#define __MAXEXP_H

MTS_NAMESPACE_BEGIN

class MaxExpDist {
public:
	MaxExpDist(const std::vector<Float> &sigmaT) 
	 : m_sigmaT(sigmaT), m_cdf(sigmaT.size()+1), m_intervalStart(sigmaT.size()) {
		/* Sort the coefficients in decreasing order */
		std::sort(m_sigmaT.begin(), m_sigmaT.end(), std::greater<Float>());

		m_cdf[0] = 0;
		for (size_t i=0; i<m_sigmaT.size(); ++i) {
			/* Integrate max(f_1(t), .., f_n(t)) on [0, \infty]*/
			Float lower = (i==0) ? -1 : -std::pow((m_sigmaT[i]/m_sigmaT[i-1]), 
						-m_sigmaT[i] / (m_sigmaT[i]-m_sigmaT[i-1]));
			Float upper = (i==m_sigmaT.size()-1) ? 0 : -std::pow((m_sigmaT[i+1]/m_sigmaT[i]), 
						-m_sigmaT[i] / (m_sigmaT[i+1]-m_sigmaT[i]));
			m_cdf[i+1] = m_cdf[i] + (upper - lower);

			/* Store the interval covered by each f_i */
			m_intervalStart[i] = (i == 0) ? 0
				: std::log(m_sigmaT[i]/m_sigmaT[i-1]) / (m_sigmaT[i]-m_sigmaT[i-1]);
		}

		/* Turn into a discrete CDF and keep the normalization factor */
		m_normalization = m_cdf[m_cdf.size()-1];
		m_invNormalization = 1 / m_normalization;

		for (size_t i=0; i<m_cdf.size(); ++i) 
			m_cdf[i] *= m_invNormalization;
	}

	Float sample(Float u, Float &pdf) const {
		/* Find the f_i for this sample */
		const Float *lowerBound = std::lower_bound(&m_cdf[0], &m_cdf[m_cdf.size()], u);
		int index = std::max(0, (int) (lowerBound - &m_cdf[0]) - 1);
		SAssert(index >= 0 && index < (int) m_sigmaT.size());

		/* Sample according to f_i */
		Float t = -std::log(std::exp(-m_intervalStart[index] * m_sigmaT[index]) 
			- m_normalization * (u - m_cdf[index])) / m_sigmaT[index];
	
		/* Compute the probability of this sample */
		pdf = m_sigmaT[index] * std::exp(-m_sigmaT[index] * t) * m_invNormalization;

		return t;
	}

	Float cdf(Float t) const {
		const Float *lowerBound = std::lower_bound(&m_intervalStart[0], 
				&m_intervalStart[m_intervalStart.size()], t);
		int index = std::max(0, (int) (lowerBound - &m_intervalStart[0]) - 1);
		SAssert(index >= 0 && index < (int) m_sigmaT.size());

		Float lower = (index==0) ? -1 : -std::pow((m_sigmaT[index]/m_sigmaT[index-1]), 
					-m_sigmaT[index] / (m_sigmaT[index]-m_sigmaT[index-1]));
		Float upper = -std::exp(-m_sigmaT[index] * t);

		return m_cdf[index] + (upper - lower) * m_invNormalization;
	}
private:
	std::vector<Float> m_sigmaT;
	std::vector<Float> m_cdf;
	std::vector<Float> m_intervalStart;
	Float m_normalization, m_invNormalization;
};

MTS_NAMESPACE_END

#endif /* __MAXEXP_H */