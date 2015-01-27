#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <execinfo.h>

#include <cuda.h>

#include "cudasampler.hpp"

#include "stats.hpp"

std::ostream &operator<<(std::ostream &os, const float2 &p)
{
  return os << '(' << p.x << ',' << p.y << ')';
}

void
cuda::cudaerror::showStack()
{
  if(!*this) {
    void* callstack[128];
    int i, frames = backtrace(callstack, 128);
    char** strs = backtrace_symbols(callstack+1, frames-1);
    for (i = 0; i < frames-1; ++i) {
      fprintf(stderr, "%s\n", strs[i]);
    }
    free(strs);
  }
}

void
cuda::syncDevice(void)
{
  cudaerror err(::cudaDeviceSynchronize());
  if (!err)
    throw err;
}

void
cuda::checkErrorsMaybeSyncDevice(void)
{
  cudaerror err(::cudaGetLastError());
  if(!err)
    err = ::cudaDeviceSynchronize ();
  if (!err)
    throw err;
}

void *
cuda::cudaMalloc(size_t nbytes)
{
  void *ret;
  cudaerror err(::cudaMalloc(&ret, nbytes));
  if (!err)
    throw err;
  return ret;
}

void
cuda::cudaFree(void *ptr)
{
  cudaerror err(::cudaFree(ptr));
  if (!err) {
    throw err;
  }
}

void
cuda::cudaFreeNoThrow(void *ptr) throw()
{
  cudaerror err(::cudaFree(ptr));
#ifndef NDEBUG
  if (!err) {
    std::cerr << "cudaFree(" << ptr << ") = '"
	      << err.what() << "'\n";
  }
#endif
}

void
cuda::cudaMemcpyDeviceToHost (void *dst, const void *src, size_t nel)
{
  cudaerror err(::cudaMemcpy(dst, src, nel, ::cudaMemcpyDeviceToHost));
  if (!err)
    throw err;
}

void
cuda::cudaMemcpyHostToDevice (void *dst, const void *src, size_t nel)
{
  cudaerror err(::cudaMemcpy(dst, src, nel, ::cudaMemcpyHostToDevice));
  if (!err)
    throw err;
}

const char *
cuda::cudaerror::what() const throw()
{
  return ::cudaGetErrorString(err);
}

/** Basic parallel accummulate.  After evaluation, the sum over @arr
 * will be left in @{arr[0]}
 */
template<typename T>
void __device__
parAccum (T *arr, int nobj)
{
  const int tid = threadIdx.x;
  for (nobj /= 2; nobj; nobj /= 2) {
    for (int i = tid; i < nobj; i += blockDim.x) {
      arr[i] += arr[i + nobj];
    }
    __syncthreads ();
  } while (nobj);
}

static __global__ void
calcprods (int nfiles, int nclus,
	   float *prod,   // [2^nfiles-1,nclus]
     const float *weight) // [nclus,nfiles]
{
  int
    tid  = threadIdx.x,
    comb = blockIdx.x * blockDim.x + tid,
    clus = blockIdx.y;

  int Ksqm1 = (1<<nfiles)-1;

  extern __shared__ float lweight[]; // nfiles
  if (tid < nfiles) {
    lweight[tid] = weight[tid * nclus + clus];
  }
  __syncthreads();

  if (comb < Ksqm1) {
    float p = 1;
    int k, kk = comb+1;
    while (k = __ffs(kk)) {
      k -= 1;
      p *= lweight[k];
      kk &= ~(1<<k);
    }
    prod[comb * nclus + clus] = p;
  }
}

template<int blocksize>
static __global__ void
colsums (int nfiles, int nclus,
	 float *prodsum, // [2^nfiles-1]
   const float *prod)   // [2^nfiles-1,nclus]
{
  int tid = threadIdx.x;
  int KK = blockIdx.x;

  float sum = 0;
  for (int i = tid; i < nclus; i += blockDim.x) {
    sum += prod[KK * nclus + i];
  }

  __shared__ float csum[blocksize];
  csum[tid] = sum;
  __syncthreads ();

  parAccum (csum, blockDim.x);

  if (tid == 0) {
    prodsum[KK] = csum[0];
  }
}

static __global__ void
logweights (const float *weight, float *lweight, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lweight[i] = log(weight[i]);
  }
}

// how the number of coefficients scales with the number of files:
//   3 =   15
//   4 =   52
//   5 =  203
//   6 =  877
//   7 = ????  never let it run this long, but should be 4 to 5k
// note that the "weight order" scales *much* faster
static __device__ void
calcProdCoef (int nfiles, int ncoef,
	      float *coef,
	const int   *weightord,
	const float *colprodsum)
{
  for (int index = threadIdx.x; index < ncoef; index += blockDim.x) {
    float prod = 1;
    for (int i = weightord[index], n = weightord[index+1];
	 i < n; i++) {
      prod *= colprodsum[weightord[i]];
    }
    coef[index] = prod;
  }
}

static __device__ void
calcProdPhi (int nfiles, int nphiord,
	     float *local,
       const float *coef,
       const float *phi,
       const int   *phiord)
{
  const int
    tid  = threadIdx.x,
    nphi = nfiles*(nfiles-1)/2;

  if (tid < nphi) {
    local[tid] = phi[tid];
  }
  __syncthreads();

  float sum = 0;
  for (int i = tid; i < nphiord; i += blockDim.x) {
    float prod = coef[phiord[i]];
    int k, kk = i;
    while (k = __ffs(kk)) {
      k -= 1;
      prod *= local[k];
      kk &= ~(1<<k);
    }
    sum += prod;
  }

  // we are reusing @local here, so need to make sure everyone else
  // has finished using it to cache phi
  __syncthreads ();

  // accumulate all into local[0]
  local[tid] = sum;
  __syncthreads ();
  parAccum (local, blockDim.x);
}

class cudaOpGammaSample {
  // init RNG
  r123::distr _rng;

public:
  __device__ cudaOpGammaSample(int ittr, int id)
    : _rng((r123::distr::ukey_type){{ittr, id}}) { }

  __device__ float operator()(float alpha, float beta) {
    return _rng.rgamma(alpha) / beta;
  }
};

struct cudaOpGammaStoreAlpha {
  __device__ cudaOpGammaStoreAlpha(int ittr, int id) {}

  __device__ float operator()(float alpha, float beta) { return alpha; }
};
struct cudaOpGammaStoreBeta {
  __device__ cudaOpGammaStoreBeta(int ittr, int id) {}

  __device__ float operator()(float alpha, float beta) { return beta; }
};

template<typename OP> __global__ void
nuConditional (int nfiles, int nitems, int ncoef, int nphiord, int ittr,
	       float *out,
	 const float *phi,
	 const float *colprodsum,
	 const int   *weightord,
	 const int   *phiord)
{
  extern __shared__ float coef[]; // [ncoef+max(nphi,blockDim.x)]
  float *local = coef + ncoef;

  calcProdCoef (nfiles, ncoef, coef, weightord, colprodsum);
  calcProdPhi (nfiles, nphiord, local, coef, phi, phiord);

  if (threadIdx.x == 0) {
    OP op(ittr, 0);
    *out = op(nitems, local[0]);
  }
}

__device__ uint2
uppertri_rc (const int i, const int M)
{
  const int
    ii = M*(M-1)/2-i,
    K = floorf((sqrtf(8*ii)-1)/2);
  return make_uint2
    (M-2-K,
     i - (M-2)*(M-1)/2 + K*(K+1)/2+1);
}

static __device__ int
phiConditionalAlpha (int nfiles, int nitems, int phiidx,
		     int *matchsum,
	       const int *alloc)
{
  const int
    tid  = threadIdx.x;

  uint2 dd = uppertri_rc(phiidx, nfiles);

  int matching = 0;
  for (int i = tid; i < nitems; i += blockDim.x) {
    matching += alloc[dd.x * nitems + i] == alloc[dd.y * nitems + i];
  }

  matchsum[tid] = matching;
  __syncthreads ();

  parAccum (matchsum, blockDim.x);

  return matchsum[0];
}

static __device__ float
phiConditionalBeta (int nfiles, int nphiord, int phiidx,
	      const float *phi,
	      const float *coef,
	      const int   *phiord,
		    void *tmp)
{
  const int
    tid  = threadIdx.x,
    nphi = nfiles*(nfiles-1)/2;

  float *phicache = (float*)tmp; // [nphi]

  if (tid < nphi) {
    phicache[tid] = phi[tid];
  }
  __syncthreads();

  float sum = 0;
  for (int i = tid; i < nphiord; i += blockDim.x) {
    if (!(i & (1<<phiidx)))
      continue;
    float prod = coef[phiord[i]];
    int k, kk = i;
    while (k = __ffs(kk)) {
      k -= 1;
      prod *= phicache[k];
      kk &= ~(1<<k);
    }
    sum += prod;
  }

  // we are reusing shared memory here, so need to make sure everyone
  // else has finished using it to cache phi
  __syncthreads ();

  float *prodsum = (float*)tmp; // [blocksize]
  prodsum[tid] = sum;

  __syncthreads ();
  parAccum (prodsum, blockDim.x);

  return prodsum[0];
}

template<typename OP> __global__ void
phiConditional (int nfiles, int nitems, int ncoef, int nphiord, int ittr,
		float *out,
          const float *phi,
          const int   *alloc,   // [nfiles,nitems]
	  const float *colprodsum,
	  const int   *weightord,
	  const int   *phiord,
	  const float *nu)
{
  const int
    tid    = threadIdx.x,
    phiidx = blockIdx.x;

  extern __shared__ float coef[];
  void *tmpptr = coef + ncoef;

  calcProdCoef (nfiles, ncoef, coef, weightord, colprodsum);
  __syncthreads ();

  const float
    alpha = 1.0 + phiConditionalAlpha(nfiles, nitems, phiidx, (int*)tmpptr, alloc),
    beta  = 0.2 + *nu * phiConditionalBeta(nfiles, nphiord, phiidx, phi, coef, phiord, tmpptr) / phi[phiidx];

  if (tid == 0) {
    OP op(ittr, phiidx);
    out[phiidx] = op(alpha, beta);
  }
}

template<typename OP> __global__ void
gammaConditional (int nfiles, int nclus, int nitems,
		  int ncoef, int nphiord, int ittr,
		  float *out,
	    const float *weight,
	    const float *phi,
	    const int   *ialloc,
	    const float *colprodsum,
	    const int   *weightord,
	    const int   *phiord,
	    const float *nu,
	    const float *dpm)
{
  const int
    clus = blockIdx.x,
    file = blockIdx.y;

  extern __shared__ float coef[]; // [ncoef+max(nphi,blockDim.x)]
  float *local = coef + ncoef;

  weight += clus;

  for (int index = threadIdx.x; index < ncoef; index += blockDim.x) {
    float prod = 1;
    for (int i = weightord[index], n = weightord[index+1];
	 i < n; i++) {
      int kk = weightord[i];
      // if this product is affected by our dataset of interest
      if ((kk+1) & (1<<file)) {
	int k, kk1 = (kk+1) & ~(1<<file);
	while (k = __ffs(kk1)) {
	  k -= 1;
	  prod *= weight[k * nclus];
	  kk1 &= ~(1<<k);
	}
      } else {
	// otherwise, just use the precalculated value
	prod *= colprodsum[kk];
      }
    }
    coef[index] = prod;
  }

  calcProdPhi (nfiles, nphiord, local, coef, phi, phiord);

  if (threadIdx.x == 0) {
    const int
      off   = file * (nclus + nitems + 1),
      start = ialloc[off+clus],
      end   = ialloc[off+clus+1];

    const float
      alpha = (dpm[file] / nclus) + 1 + (end - start),
      beta  = 1.0 + *nu * local[0];

    OP op(ittr, file * nclus + clus);
    out[file * nclus + clus] = op(alpha, beta);
  }
}

class loggammapdf {
  float shapem1, lrgam;
public:
  __device__ loggammapdf(float shape) : shapem1(shape-1), lrgam(-lgammaf(shape)) {}

  __device__ float operator()(float x) { return lrgam + __logf(x)*shapem1 - x; }
};

static __device__ float
gammaLogPdf (float x, float shape, float rate)
{
  float ratecorr = 0;
  if (rate != 1.f) {
    ratecorr = __logf(rate)*shape - (rate*x);
  }
  return -lgammaf(shape) + __logf(x)*(shape-1) + ratecorr;
}

template<int blocksize>
static __global__ void
dpMassSample (int nfiles, int nclus, int ittr,
	      float *dpm,
	const float *weight)
{
  int
    tid  = threadIdx.x,
    file = blockIdx.x;

  __shared__ float dp1, dp2, runif;
  float ll = 0;
  if(tid  == 0) {
    // init RNG
    r123::distr rng((r123::distr::ukey_type){{ittr, file}});

    dp1 = dpm[file];
    dp2 = abs(dp1 + rng.rnorm() * 1.f);

    runif = rng.runif();

    ll +=
      +gammaLogPdf(dp1, 2, 4)
      -gammaLogPdf(dp2, 2, 4);
  }
  __syncthreads();

  const float
    shape1 = dp1 / nclus,
    shape2 = dp2 / nclus;

  weight += file * nclus;
  for (int clus = tid; clus < nclus; clus += blockDim.x) {
    ll +=
      +gammaLogPdf(weight[clus], shape1, 1.f)
      -gammaLogPdf(weight[clus], shape2, 1.f);
  }

  __shared__ float llsum[blocksize];
  llsum[tid] = ll;
  __syncthreads();
  parAccum (llsum, blockDim.x);

  if (tid == 0) {
    if (llsum[0] < -__logf(runif)) {
      dpm[file] = dp2;
    }
  }
}

cuda::sampler::sampler (int K, int n, int N,
      const std::vector<int> &phiord,
      const std::vector<int> &weightord)
  : _nfiles(K), _nitems(n), _nclus(N),
    _ittr(0), _nsamplers(0)
{
  const int
    Ksqm1 = (1<<K)-1,
    nphi = K*(K-1)/2;

  d_phi.resize(nphi);
  d_lrho.resize(nphi);
  d_weight.resize(N*K);
  d_lweight.resize(N*K);
  d_alloc.resize(n*K);
  d_ialloc.resize((N+n+1)*K);
  d_prod.resize(N*Ksqm1);
  d_prodsums.resize(Ksqm1);
  d_dpm.resize(K);

  d_phiord.resize(phiord.size());
  d_phiord = phiord;

  _nweightcoef = weightord[0];
  std::vector<int> weightord2(_nweightcoef+1);
  for (int i = 0, j = 1; i < _nweightcoef; i++) {
    weightord2[i] = weightord2.size();
    int n = weightord[j++];
    for (; n > 0; n--) {
      weightord2.push_back(weightord[j++]);
    }
  }
  weightord2[_nweightcoef] = weightord2.size();

  d_weightord.resize(weightord2.size());
  d_weightord = weightord2;

  d_nu.resize(1);

  d_prob.resize(K*N*n);
}

__global__ void
phi_to_lrho(float *phi, float *lrho, int nphi)
{
  const int
    i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nphi) {
    lrho[i] = log1p(phi[i]);
  }
}

static std::vector<int>
inverseAlloc(int nitems, int nclus, int nfiles,
	     const std::vector<int> &alloc)
{
  int elems = nclus + nitems + 1;
  std::vector<int> ialloc;
  for (int file = 0; file < nfiles; file++) {
    std::vector<int> tmp(elems);

    int cur = nclus;
    tmp[cur++] = elems;

    for (int clus = 0; clus < nclus; clus++) {
      tmp[clus] = cur;

      for (int i = 0; i < nitems; i++) {
	if (alloc[file * nitems + i] == clus)
	  tmp[cur++] = i;
      }
    }
    assert(cur == elems);
    ialloc.insert(ialloc.end(), tmp.begin(), tmp.end());
  }

  if(0) {
    for (int file = 0; file < nfiles; file++) {
      std::cerr << "ialloc(" << file << ") = [";
      for (int clus = 0; clus < nclus; clus++) {
	std::cerr << (clus ? "," : "") << alloc[file * nitems + clus];
      }
      std::cerr << "] = {\n";
      for (int clus = 0; clus < nclus; clus++) {
	std::cerr << "  " << clus << " = [";
	for (int
	       cur = ialloc[elems * file + clus],
	       end = ialloc[elems * file + clus + 1]; cur < end; cur++) {
	  std::cerr << ialloc[elems * file + cur] << (cur + 1 < end ? " " : "");
	}
	std::cerr << "]\n";
      }
      std::cerr << "}\n";
    }
  }

  return ialloc;
}

void
cuda::sampler::allocChanged(const std::vector<int> &alloc)
{
  d_ialloc = inverseAlloc(_nitems, _nclus, _nfiles, alloc);
}

void
cuda::sampler::setAlloc(const std::vector<int> &alloc)
{
  d_alloc  = alloc;
  allocChanged(alloc);
}

void
cuda::sampler::setPhis(const std::vector<float> &phi)
{
  const int
    nphi  = _nfiles*(_nfiles-1)/2;

  if (nphi > 0) {
    d_phi = phi;
    phi_to_lrho <<<(nphi+31)/32,32>>> (d_phi, d_lrho, nphi);
  }
}

void
cuda::sampler::setWeights(const std::vector<float> &weight)
{
  const int
    Ksqm1 = (1<<_nfiles)-1;

  d_weight = weight;

  dim3 Dg(1, _nclus), Db(Ksqm1);
  if (Ksqm1 >= 32) {
    Dg.x = (Ksqm1+31)/32;
    Db.x = 32;
  }
  calcprods<<<Dg,Db,sizeof(float)*_nfiles>>>
    (_nfiles, _nclus, d_prod, d_weight);

  colsums<32><<<Ksqm1,32>>>(_nfiles, _nclus, d_prodsums, d_prod);

  int n = _nfiles * _nclus;
  logweights<<<(n+15)/16,16>>>(d_weight, d_lweight, n);
}

template<typename OP> void
cuda::sampler::runNuConditional (float *out, int ittr) const
{
  const int
    nphi  = _nfiles*(_nfiles-1)/2,
    ncoef = _nweightcoef;

  nuConditional
    <OP>
    <<<1,16,sizeof(float)*(ncoef+max(nphi,16))>>>
    (_nfiles, _nitems, ncoef, d_phiord.size(), ittr,
     out, d_phi, d_prodsums, d_weightord, d_phiord);
  checkErrorsMaybeSyncDevice();
  syncDevice();
}

void
cuda::sampler::sampleNuConditional ()
{
  runNuConditional<cudaOpGammaSample>(d_nu, _ittr++);
}

float
cuda::sampler::collectNuConditionalAlpha () const
{
  vector<float> nuout(1);
  runNuConditional<cudaOpGammaStoreAlpha>(nuout, 0);
  return nuout.get(0);
}

float
cuda::sampler::collectNuConditionalBeta () const
{
  vector<float> nuout(1);
  runNuConditional<cudaOpGammaStoreBeta>(nuout, 0);
  return nuout.get(0);
}

template<typename OP> void
cuda::sampler::runPhiConditionals (float *out, int ittr) const
{
  const int
    nphi  = _nfiles*(_nfiles-1)/2,
    ncoef = _nweightcoef;

  // executing with a size of zero seems to give strange errors, so
  // check for this here...
  if (nphi > 0) {
    phiConditional
      <OP>
      <<<nphi, 16, sizeof(float)*(ncoef+max(nphi,16))>>>
      (_nfiles, _nitems, ncoef, d_phiord.size(), ittr,
       out, d_phi, d_alloc, d_prodsums, d_weightord, d_phiord, d_nu);
    syncDevice();
  }
}

void
cuda::sampler::samplePhiConditionals ()
{
  runPhiConditionals<cudaOpGammaSample>(d_phi, _ittr++);
}

std::vector<float>
cuda::sampler::collectPhiConditionalsAlpha () const
{
  vector<float> out(d_phi.size());
  runPhiConditionals<cudaOpGammaStoreAlpha>(out, 0);
  return out;
}

std::vector<float>
cuda::sampler::collectPhiConditionalsBeta () const
{
  vector<float> out(d_phi.size());
  runPhiConditionals<cudaOpGammaStoreBeta>(out, 0);
  return out;
}

template<typename OP> void
cuda::sampler::runGammaConditionals (float *out, int ittr) const
{
  const int
    nphi  = _nfiles*(_nfiles-1)/2,
    ncoef = _nweightcoef;

  gammaConditional
    <OP>
    <<<dim3(_nclus,_nfiles), 16, sizeof(float)*(ncoef+max(nphi,16))>>>
    (_nfiles, _nclus, _nitems, ncoef, d_phiord.size(), ittr,
     out, d_weight, d_phi, d_ialloc, d_prodsums, d_weightord, d_phiord, d_nu, d_dpm);
  syncDevice();
}

void
cuda::sampler::sampleGammaConditionals ()
{
  runGammaConditionals<cudaOpGammaSample>(d_weight, _ittr++);
}

std::vector<float>
cuda::sampler::collectGammaConditionalsBeta () const
{
  vector<float> out(_nfiles * _nclus);
  runGammaConditionals<cudaOpGammaStoreBeta>(out, 0);
  return out;
}

void
cuda::sampler::sampleDpMass ()
{
  dpMassSample<16>
    <<<_nfiles, 16>>>
    (_nfiles, _nclus, _ittr++, d_dpm, d_weight);
  checkErrorsMaybeSyncDevice();
  syncDevice();
}

cuda::sampler::~sampler()
{
  // while (_datasets.size() > 0) {
  //   delete _datasets.back();
  //   _datasets.pop_back();
  // }
}

class cuda::dataset*
cuda::sampler::nextFileSampler ()
{
  return new dataset(this, _nsamplers++);
}
