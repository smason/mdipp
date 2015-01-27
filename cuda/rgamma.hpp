#include "boxmuller.hpp"

namespace r123 {

R123_CUDA_DEVICE R123_STATIC_INLINE float pow2(float f) { return f*f; }
R123_CUDA_DEVICE R123_STATIC_INLINE float pow3(float f) { return f*f*f; }
R123_CUDA_DEVICE R123_STATIC_INLINE float pow4(float f) { f *= f; return f*f; }

// this method (from R, GPL) is only valid when $\alpha < 1$
R123_CUDA_DEVICE R123_STATIC_INLINE bool
rgamma_gs (uint32_t u0, uint32_t u1,
	   float alpha,
	   float *out)
{
  const float
    EXP_M1f = 0.36787944117144232159f,
    e = 1.0 + EXP_M1f * alpha;
  float
    p1 = e * u01<float>(u0),
    p2 = -__logf(u01<float>(u1));
  if (p1 >= 1.0) {
    float x = -__logf((e - p1) / alpha);
    if (p2 >= (1.0 - alpha) * __logf(x)) {
      *out = x;
      return true;
    }
  } else {
    float x = powf(p1, 1/alpha);
    if (p2 >= x) {
      *out = x;
      return true;
    }
  }
  return false;
}

// Marsaglia and Tsang's method is only valid when $\alpha >= 1$
R123_CUDA_DEVICE R123_STATIC_INLINE bool
rgamma_mt (float X, uint32_t U,
	   float alpha,
	   float *out)
{
  const float
    d = alpha - (1.f/3.f),
    c = (1.f/3.f) / sqrtf(d),
    v = pow3(1 + c * X);

  // see if the sample is acceptable
  if (v > 0 &&
      __logf(u01<float>(U)) < 0.5*pow2(X) + d*(1-v+__logf(v))) {
    *out = d*v;
    return true;
  }
  return false;
}

R123_CUDA_DEVICE R123_STATIC_INLINE bool
rgamma (uint32_t u0, uint32_t u1, uint32_t u2, uint32_t u3,
	float alpha,
	float *out)
{
  // check parameters are sensible!
  if (!alpha > 0) {
    if(alpha == 0) {
      *out = 0;
      return true;
    } else {
      *out = NAN;
      return true;
    }
  }
  if (alpha < 1) { /* GS algorithm for parameters a < 1 */
    if (rgamma_gs (u0, u1, alpha, out))
      return true;
    return rgamma_gs (u2, u3, alpha, out);
  }
  const float2 f2 = __boxmuller(u0, u1);
  if (rgamma_mt(f2.x, u01<float>(u2), alpha, out))
    return true;
  return rgamma_mt (f2.y, u01<float>(u3), alpha, out);
}

R123_CUDA_DEVICE R123_STATIC_INLINE bool
rnormgam (uint32_t u0, uint32_t u1, uint32_t u2, uint32_t u3,
	  float mu, float kappa, float alpha, float beta,
	  float *omu, float *otau)
{
  float2 f2 = __boxmuller(u0, u1);

  float taub;
  if (alpha < 1) {
    if (!rgamma_gs (u2, u3, alpha, &taub))
      return false;
  } else {
    if (!rgamma_mt (f2.x, u2, alpha, &taub))
      return false;
  }

  taub /= beta;
  *otau = taub;
  *omu  = f2.y / sqrt(kappa + taub);
  return true;
}

}
