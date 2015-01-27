// #include <math.h>

// static inline void sincosf(float x, float *s, float *c)   { __sincosf(x, s, c); }
// static inline void sincos(double x, double *s, double *c) { __sincos(x, s, c); }

#ifndef samsrng__hpp
#define samsrng__hpp

#include <Random123/philox.h>
#include "boxmuller.hpp"

namespace r123 {

  static inline __device__ float pow2(float f) { return f*f; }
  static inline __device__ float pow3(float f) { return f*f*f; }

class distr {
public:
  typedef Philox4x32          rng_type;
  typedef rng_type::ukey_type ukey_type;
  typedef rng_type::ctr_type  ctr_type;

private:
  ukey_type _k;
  ctr_type  _ctr, _val;
  int _cur;
  float _gauss;

  __device__ uint32_t next1() {
    if (_cur < 4) {
      return _val[_cur++];
    } else {
      _ctr.incr();
      _val = rng_type()(_ctr,_k);
      _cur = 1;
      return _val[0];
    }
  }

  __device__ uint2 next2() {
    uint32_t a, b;
    // somewhat awkward so GPU can run through in parallel
    if (_cur < 4) a = _val[_cur];
    if (_cur < 3) {
      b = _val[_cur+1];
      _cur += 2;
    } else {
      _ctr.incr();
      _val = rng_type()(_ctr,_k);
      if (_cur > 3) {
	a = _val[0];
	b = _val[1];
	_cur = 2;
      } else {
	b = _val[0];
	_cur = 1;
      }
    }
    return make_uint2(a,b);
  }

public:
  __device__ distr(const ukey_type &k,
		   const ctr_type  &ctr = ctr_type())
    : _k(k), _ctr(ctr)
  {
    _val   = rng_type()(_ctr, _k);
    _cur   = 0;
    _gauss = NAN;
  }

  /* uniformly distributed over 0<x<=1 */
  inline __device__ float runif() {
    return u01<float>(next1());
  }
  inline __device__ float2 runif2() {
    uint2 u = next2();
    return (float2){u01<float>(u.x), u01<float>(u.y)};
  }

  inline __device__ float2 rnorm2() {
    uint2 u = next2();
    return __boxmuller(u.x,u.y);
  }
  /* standard normal, zero mean, unit variance */
  inline __device__ float rnorm() {
    float f = _gauss;
    if (!isnan(f)) {
      _gauss = NAN;
      return f;
    } else {
      float2 f2 = rnorm2();
      _gauss = f2.x;
      return f2.y;
    }
  }

  /* gamma distribution with alpha=shape paramater, divide or multiply by beta
   * as appropriate for your parameterisation */
  __device__ float rgamma(float alpha) {
    // this identity is valid $\forall \alpha$; it would be nice if I
    // could use this identity to reduce divergence between warps
    //
    // 1. maybe something could be done by tracking rejections?
    //
    // only obvious way to fix this seems to be consuming more
    // numbers; maybe I could enumerate the cases "don't matter"
    //
    // 
    float corr = 1;
    if (alpha < 1) {
      corr = powf(runif(), 1/alpha);
      alpha += 1;
    }
    const float
      d = alpha - (1.f/3.f),
      c = (1.f/3.f) / sqrtf(d);
    float X, v, U;
    while (1) {
      X = rnorm ();
      v = pow3(1 + c * X);
      if (v > 0) {
	U = runif();
	if (logf(U) < 0.5*pow2(X) + d*(1-v+logf(v)))
	  return d*v*corr;
      }
    }
  }

  /* normal-gamma distribution, mu=mean, kappa=precision, alpha=shape
   * and beta=scale.  ret.x = mean, ret.y = precision */
  inline __device__ float2 rnormgam(float mu, float kappa, float alpha, float beta) {
    float2 f;
    f.y = rgamma(alpha)/beta;
    f.x = mu + rnorm() / sqrt(kappa + f.y);
    return f;
  }
};

}

#endif
