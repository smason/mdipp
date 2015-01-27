#ifndef mdi_stats_hpp
#define mdi_stats_hpp

#include <cmath>

#ifndef __CUDACC__
#include <iostream>
#include <random>

extern std::default_random_engine generator;

#define HOSTORDEVICE
#else
#define HOSTORDEVICE __host__ __device__
#include "cuda/samsrng.hpp"
#endif // #ifndef __CUDACC__

#ifndef __CUDACC__
// whether we should accept a MCMC step with the given log-likelihood ratio
template<class T = double>
bool mcmcAcceptProposalLogProb (T llratio) {
  return llratio < std::exponential_distribution<>()(generator);
}

// as above, but seperate out the components so we can reorder the
// expression to do more senisble things with infinities
template<class T = double>
bool mcmcAcceptProposalLogProb (T llcur, T llprop) {
  return llcur - std::exponential_distribution<>()(generator) < llprop;
}
#endif

// wrap the std::gamma_distribution type in something more useful for doing Bayesian stats
template<class T = double>
class gamma_distribution {
  T _shape, _rate;

public:
  typedef T result_type;

  gamma_distribution(result_type shape = 1.0, result_type rate = 1.0)
   : _shape(shape), _rate(rate) {}

  gamma_distribution(const gamma_distribution<result_type> &other)
   : _shape(other._shape), _rate(other._rate) {}

  result_type shape() const { return _shape; }
  result_type rate()  const { return _rate; }

#ifndef __CUDACC__
  gamma_distribution(const std::gamma_distribution<result_type> &other)
   : _shape(other.alpha()), _rate(1.0/other.beta()) {}

  operator std::gamma_distribution<result_type>() const {
    return std::gamma_distribution<result_type>(_shape, 1.0/_rate);
  }

  template<class URNG>
  result_type operator()(URNG &rng) const {
    return ((std::gamma_distribution<result_type>)*this)(rng);
  }
#endif

  result_type pdf(result_type x) const { return exp(logpdf(x)); }
  result_type logpdf(result_type x) const {
    if (_shape < 0 || _rate <= 0)
      return NAN;
    if (x < 0)
      return -std::numeric_limits<double>::infinity();
    if (_shape == 0)
      return x == 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    if (x == 0) {
      return _shape <= 1 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }
    if(_rate == 1) {
      return -lgamma(_shape) + std::log(x)*(_shape-1);
    }
    return std::log(_rate)*_shape - lgamma(_shape) + std::log(x)*(_shape-1) - (_rate*x);
  };
};

template<class T = double>
class gaussian_distribution {
  T _sigma;

public:
  typedef T result_type;

  gaussian_distribution(result_type sigma) : _sigma(sigma) { }

  result_type sigma()     const { return _sigma; }
  result_type variance()  const { return _sigma * _sigma; }
  result_type precision() const { return 1.0/variance(); }

#ifndef __CUDACC__
  operator std::normal_distribution<result_type>() const {
    return std::normal_distribution<result_type>(0, _sigma);
  }

  template<class URNG>
  result_type operator()(URNG &rng) const {
    return ((std::normal_distribution<result_type>)*this)(rng);
  }
#endif

#define LOG_SQRT_2PI 0.9189385332046727417803297364056176398613

  result_type pdf(result_type x) const { return exp(logpdf(x)); }
  result_type logpdf(result_type x) const {
    return -(log(_sigma)+LOG_SQRT_2PI) + (-x*x / (2*_sigma*_sigma));
  }
};

template<typename T>
HOSTORDEVICE bool isApprox(const T &a, const T &b, T prec=1e-8)
{
  using std::min;
  using std::abs;
  return abs(a - b) <= min(abs(a), abs(b)) * prec;
}

// see http://www.johndcook.com/standard_deviation.html
// and http://www.johndcook.com/skewness_kurtosis.html
template<typename T=double>
class runningstats {
  int _n;
  T _m, _s;

public:
  static const runningstats zero;

  HOSTORDEVICE runningstats() : _n(0), _m(0), _s(0) { }
  HOSTORDEVICE runningstats(int n, T m, T s) : _n(n), _m(m), _s(s) { }
  HOSTORDEVICE runningstats(const runningstats &x) : _n(x._n), _m(x._m), _s(x._s) { }

  HOSTORDEVICE runningstats& operator=(const runningstats &x) {
    _n = x._n;
    _m = x._m;
    _s = x._s;
    return *this;
  }

  HOSTORDEVICE runningstats& operator+=(const runningstats& rhs) {
    if (rhs._n == 0)
      return *this;
    if (_n == 0)
      return *this = rhs;

    int n = _n + rhs._n;
    T
      d1 = _m - rhs._m,
      d2 = d1*d1,
      m  = (_n*_m + rhs._n*rhs._m) / n,
      s  = _s + rhs._s + d2 * _n * rhs._n / n;

    _n = n;
    _m = m;
    _s = s;

    return *this;
  }

  HOSTORDEVICE bool operator==(const runningstats &x) const {
    if (_n != x._n)
      return false;
    if (_n == 0)
      return true;
    std::cout << (_m - x._m) << ' ' << (_s - x._s) << '\n';
    return _m == x._m && _s == x._s;
  }

  HOSTORDEVICE bool isApprox(const runningstats &x, T prec=1e-6) const {
    if (_n != x._n)
      return false;
    if (_n == 0)
      return true;
    return ::isApprox(_m, x._m, prec) && ::isApprox(_s, x._s, prec);
  }

  HOSTORDEVICE void reset() { _n = 0; }

  HOSTORDEVICE void push(T x) {
    if (++_n == 1) {
      _m = x;
      _s = 0;
    } else {
      const T dx = x - _m;
      _m += dx/_n;
      _s += dx*(x - _m);
    }
  }

  HOSTORDEVICE void pop(T x) {
    _n -= 1;
    assert (_n > 0);

    const T dx = x - _m;
    _m -= dx/_n;
    _s -= dx*(x - _m);
  }

  HOSTORDEVICE int count() const { return _n; };

  HOSTORDEVICE T mean()  const { return _m; }
  HOSTORDEVICE T sumsq() const { return _s; }
  HOSTORDEVICE T var()   const { return _s/(_n-1); }
  HOSTORDEVICE T sd()    const { return std::sqrt(var()); }
};

template<typename T>
inline runningstats<T> operator+(runningstats<T> a, const runningstats<T> &b) {
  a += b;
  return a;
}

double lndt_alt (double x, double nu, double lambda);

// normal distribution with a normal-gamma conjugate prior, i.e. a normal prior on the mean and a gamma on the precision
template<typename T=double>
class normgam {
  T mu, kappa, alpha, beta;
public:
  HOSTORDEVICE
  normgam(T mu, T kappa,
          T alpha, T beta) :
    mu(mu), kappa(kappa),
    alpha(alpha), beta(beta) {
  }

  T pdf (T x, T tau) const { return exp(logpdf(x, tau)); }

  T logpdf (T x, T tau) const {
    const T xmu = x-mu;
    return (log(beta)*alpha + log(kappa)*0.5) - (lgamma(alpha) + log(2*M_PI)*0.5) +
      log(tau)*(alpha-0.5) - (beta*tau) - (kappa*tau*xmu*xmu*0.5);
  }

  // Posterior predictive of observing @m new data points
  T lnppredict(normgam D, unsigned int m) const;

  // Posterior predictive of observing a single @x given current hyperparameters
  //  See Equation 100 from "Conjugate Bayesian analysis of the Gaussian distribution" by Kevin Murphy 2007
  T lnppredict(T x) const {
    return lndt_alt(x-mu, 2.*alpha, alpha*kappa/(beta*(kappa+1.)));
  }

  // update hyperparameters with given normal
  HOSTORDEVICE normgam update (unsigned int n, T mean, T sumsq) const {
    const T
      mmu    = mean-mu,
      kappan = kappa + n;
    if(n == 0)
      return *this;
    return normgam<T>((kappa * mu + n * mean) / kappan,
		      kappan,
		      alpha + n/2.,
		      beta + sumsq/2. + mmu*mmu/2. * (kappa*n) / kappan);
  }

  // helper for use with @runningstats
  HOSTORDEVICE normgam update (const runningstats<T> &rs) const {
    return update(rs.count(), rs.mean(), rs.sumsq());
  }

#ifndef __CUDACC__
  // returns a pair with @first = mean, @second = precision (i.e. 1/variance)
  template<typename URNG>
  std::pair<T,T> sample (URNG &g) const {
    T
      tau = (gamma_distribution<T>(alpha, beta))(g),
      x   = (std::normal_distribution<T>(mu, 1.0/sqrt(kappa+tau)))(g);
    return std::make_pair(x, tau);
  }
#else
  __device__ float2 cudaSample (r123::distr &d) const {
    return d.rnormgam(mu, kappa, alpha, beta);
  }
#endif
};

#ifndef __CUDACC__
// hacked together from:
//   http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
namespace Eigen {
  namespace internal {
    template<typename Scalar>
    struct scalar_normal_dist_op {
      EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

      template<typename Index>
      inline const Scalar operator() (Index, Index = 0) const {
        return std::normal_distribution<Scalar>()(generator);
      }
    };

    template<typename Scalar>
    struct functor_traits<scalar_normal_dist_op<Scalar> >
    { enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };
  } // end namespace internal
} // end namespace Eigen

template<class T = double, int _Size = Eigen::Dynamic>
class multivariatenormal_distribution {
public:
  typedef Eigen::Matrix<T, _Size, 1>     result_type;
  typedef Eigen::Matrix<T, _Size, _Size> covariance_type;

private:
  covariance_type _cov;

public:
  multivariatenormal_distribution(const covariance_type &cov) : _cov(cov) { }

  template<class URNG>
  result_type operator()(URNG &rng) const {
    Eigen::LLT<covariance_type> cholSolver(_cov);

    covariance_type normTransform;
    // We can only use the cholesky decomposition if
    // the covariance matrix is symmetric, pos-definite.
    // But a covariance matrix might be pos-semi-definite.
    // In that case, we'll go to an EigenSolver
    if (cholSolver.info() == Eigen::Success) {
      // Use cholesky solver
      normTransform = cholSolver.matrixL();
    } else {
      std::cerr << "multivariatenormal_distribution: using the eigen solver\n";
      // Use eigen solver
      Eigen::SelfAdjointEigenSolver<covariance_type> eigenSolver(_cov);
      normTransform = eigenSolver.eigenvectors()
        * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::internal::scalar_normal_dist_op<T> randN; // Gaussian functor

    return normTransform * result_type::NullaryExpr(_cov.cols(), randN);
  }

#define LOG_2_PI 1.8378770664093454835606594728112352797226

  T pdf(result_type x) const { return exp(logpdf(x)); }
  T logpdf(result_type x) const {
    Eigen::LLT<covariance_type> cholSolver(_cov);

    covariance_type covinv;
    T logdet;

    // We can only use the cholesky decomposition if
    // the covariance matrix is symmetric, pos-definite.
    // But a covariance matrix might be pos-semi-definite.
    // In that case, we'll go to an EigenSolver
    if (cholSolver.info() == Eigen::Success) {
      // Use cholesky solver
      covinv = cholSolver.solve(Eigen::MatrixXd::Identity(_cov.rows(),_cov.cols()));
      covariance_type lm = cholSolver.matrixL();
      logdet = lm.diagonal().array().log().sum() * 2;
    } else {
      std::cerr << "multivariatenormal_distribution: cholesky decomposition failed\n";
      covinv = _cov.inverse();
      logdet = -covinv.householderQr().logAbsDeterminant();
    }

    const T distval = ((x.transpose() * covinv).transpose().array() * x.array()).sum();
    return -(x.size() * LOG_2_PI + logdet + distval)/2;
  }
};
#endif // #ifndef __CUDACC__

#endif // #ifndef mdi_stats_hpp
