#include <cuda.h>

#include <stdio.h>
#include <assert.h>

#include "cudasampler.hpp"

#include "stats.hpp"

/** would be good to have something like this, but nvcc doesn't
 * support C++11 lambdas yet so this just complicates things
 */
template<typename OP>
__device__ void
ittrAlloc(OP op, const int *ialloc, int file, int clus, int nclus)
{
  for (const int
	 *cur = ialloc + ialloc[clus],
	 *end = ialloc + ialloc[clus+1];
       cur < end;
       cur++) {
    op(*cur);
  }
}

// (x,y) = (cluster,feature)
template<typename OP> __global__ void
gaussian_samplepars (int nitems, int nclus, int nfeatures, int file,
               const float*  x,      // [nfeatures,nitems]
               const int*    ialloc, // [nclus+nitems+1]
                     float2* par,    // [nfeatures,nclus]
                     int ittr)
{
  const int
    clus = blockIdx.x * blockDim.x + threadIdx.x,
    feat = blockIdx.y;

  if (clus < nclus) {
    x   += feat * nitems;
    par += feat * nclus + clus;

    runningstats<float> rs;
    for (const int
	   *cur = ialloc + ialloc[clus],
 	   *end = ialloc + ialloc[clus+1];
	 cur < end;
	 cur++) {
      rs.push(x[*cur]);
    }

    OP op(ittr, clus * nfeatures + feat);
    *par = op(rs);
  }
}

// when we want a sample back
class gaussian_samplepars_gensample {
  r123::distr _rng;

public:
  __device__ gaussian_samplepars_gensample(int ittr, int id)
    : _rng((r123::distr::ukey_type){{ittr, id}}) {};

  __device__ float2 operator()(const runningstats<float> &rs) {
    return normgam<float>(0, 1, 2, 2).update(rs).cudaSample(_rng);
  }
};

// for testing, we want to see whether the calculated mean and variance are OK
struct gaussian_samplepars_meansd {
  __device__ gaussian_samplepars_meansd(int ittr, int id) {}

  __device__ float2 operator()(const runningstats<float> &rs) {
    return make_float2(rs.mean(), rs.sd());
  }
};

template<int blocksize>
__global__ void
gaussian_accumprobs (int nitems, int nclus, int nfeatures,
               const float*  x,     // [nfeatures,nitems]
               const float2* par,   // [nfeatures,nclus]
                     float*  lprob) // [nclus,nitems]
{
  if(blocksize != blockDim.x) asm("trap;");

  int
    tid  = threadIdx.x,
    clus = blockIdx.x * blocksize + tid,
    item = blockIdx.y * blocksize + tid;

  x     += item;
  par   += clus;
  lprob += item;

  float sum[blocksize] = { 0 };
  for (int feat = 0; feat < nfeatures; feat++) {
    __shared__ float2 parblock[blocksize];

    // Load in the parameter values for our thread's cluster into shared memory
    if (clus < nclus)
      parblock[tid] = par[feat * nclus];

    // ensure all of this block's threads have made it up to here
    __syncthreads();

    if (item < nitems) {
      // get our thread items's feature value f
      const float xf = x[feat * nitems];
      // so it can be tested over all i cluster values
      for (int i = 0; i < blocksize; i++) {
	const float x = xf - parblock[i].x;
	sum[i] += 0.5f*__logf(parblock[i].y) - (x*x * (0.5f*parblock[i].y));
      }
    }
  }

  if (item < nitems) {
    if (0) {
      printf("*** P(%3i,%3i..%3i) = "
	     "[%8g,%8g,%8g,%8g,"
	     " %8g,%8g,%8g,%8g]\n",
	     item, blockIdx.x * blocksize, (blockIdx.x+1) * blocksize-1,
	     sum[ 0], sum[ 1], sum[ 2], sum[ 3],
	     sum[ 4], sum[ 5], sum[ 6], sum[ 7]);
    }
    for (int i = 0; i < blocksize; i++) {
      clus = blockIdx.x * blocksize + i;
      if (clus < nclus)
	lprob[clus * nitems] = sum[i];
    }
  }
}

void
cuda::gaussian::sampleParameters()
{
  const int
    blocksize  = 16,
    blockclus  = (gpu->nclus()+blocksize-1)/blocksize,
    blockitems = (gpu->nitems()+blocksize-1)/blocksize;

  gaussian_samplepars
    <gaussian_samplepars_gensample>
    <<<dim3(blockclus,nfeatures),blocksize>>>
    (gpu->nitems(), gpu->nclus(), nfeatures, gpu->myfile(), d_data,
     gpu->mygpuinverseallocs(), d_par, gpu->ittr());
}

void
cuda::gaussian::accumAllocProbs()
{
  const int
    blocksize  = 8,
    blockclus  = (gpu->nclus()+blocksize-1)/blocksize,
    blockitems = (gpu->nitems()+blocksize-1)/blocksize;

  gaussian_accumprobs<blocksize><<<dim3(blockclus,blockitems),blocksize>>>
    (gpu->nitems(), gpu->nclus(), nfeatures, d_data, d_par,
     gpu->mygpuprobs());
}

template<int blocksize>
__global__ void
gaussianprocess_accumprobs (int nitems, int nclus, int nfeatures,
		      const float*  x,       // [nfeatures,nitems]
		      const float*  sampfn,  // [nfeatures,nclus]
		      const float*  sigma,   // [nclus]
			    float*  lprob)   // [nclus,nitems]
{
  if(blocksize != blockDim.x) asm("trap;");

  int
    tid  = threadIdx.x,
    clus = blockIdx.x * blocksize + tid,
    item = blockIdx.y * blocksize + tid;

  x     += item;
  lprob += item;

  extern __shared__ float gppars[]; // [blocksize*(2+nfeatures)]
  float
    *leps   = gppars,
    *lsamfn = leps+blocksize;
  // Load in the parameter values for our thread's cluster into shared memory
  if (clus < nclus) {
    leps[tid] = sigma[tid];
  }

  float sum[blocksize] = { 0 };
  for (int feat = 0; feat < nfeatures; feat++) {
    // Load in the parameter values for our thread's cluster into shared memory
    lsamfn[tid] = sampfn[feat * nclus + clus];

    // ensure all of this block's threads have made it up to here
    __syncthreads();

    if (item < nitems) {
      // get our thread items's feature value f
      const float xf = x[feat * nitems];
      // so it can be tested over all i cluster values
      for (int i = 0; i < blocksize; i++) {
	const float x = xf - lsamfn[i];
	sum[i] += x*x;
      }
    }
  }

  if (item < nitems) {
    for (int i = 0; i < blocksize; i++) {
      clus = blockIdx.x * blocksize + i;
      if (clus < nclus)
	lprob[clus * nitems] = nfeatures*__logf(leps[i])*0.5f - sum[i]*leps[i]*0.5f;
    }
  }
}

void
cuda::gaussianprocess::accumAllocProbs ()
{
  const int
    blocksize  = 16,
    blockclus  = (gpu->nclus()+blocksize-1)/blocksize,
    blockitems = (gpu->nitems()+blocksize-1)/blocksize;

  gaussianprocess_accumprobs
    <blocksize>
    <<<dim3(blockclus,blockitems),blocksize,sizeof(float)*blocksize*(2+nfeatures)>>>
    (gpu->nitems(), gpu->nclus(), nfeatures, d_data, d_sampfn, d_sigma,
     gpu->mygpuprobs());
}

// work in "blocksize" groups of clusters for a given feature
__global__ void
multinom_samplepars (int nitems, int nclus, int nfeatures, int nlevels, int file,
               const int*   x,     // [nfeatures,nitems]
               const int*   ialloc, // [nitems+nclus+1]
                     float* lpar,  // [nfeatures,nclus,nlevels]
                     int ittr)
{
  const int
    clus = blockIdx.x * blockDim.x + threadIdx.x,
    feat = blockIdx.y;

  extern __shared__ float allstate[]; // [clus=blockDim.x,nlevels]

  // if this is a "real" cluster
  if (clus < nclus) {

    // assume float and int are same size!
    int   *statei = (int*)allstate + threadIdx.x * nlevels;
    float *statef = allstate + threadIdx.x * nlevels;
    x     += feat * nitems;
    lpar  += (feat * nclus + clus) * nlevels;

    // initialise
    for (int lev = 0; lev < nlevels; lev++)
      statei[lev] = 0;

    // add in counts
    for (const int
	   *cur = ialloc + ialloc[clus],
	   *end = ialloc + ialloc[clus+1];
	 cur < end;
	 cur++) {
      statei[x[*cur]] += 1;
    }

    // init RNG
    r123::distr rng((r123::distr::ukey_type){{ittr,clus * nfeatures + feat}});

    // multinomial draw
    float sum = 0;
    for (int lev = 0; lev < nlevels; lev++)
      sum += statef[lev] = rng.rgamma(0.5f + statei[lev]);

    // normalise this sample
    for (int lev = 0; lev < nlevels; lev++)
      lpar[lev] = __logf(statef[lev] / sum);
  }
}

// iterate over clusters=blockid.x and items=blockid.y
// blocksize, ranges over both clusters and items
template<int blocksize>
__global__ void
multinom_accumprobs (int nitems, int nclus, int nfeatures, int nlevels,
               const int*   x,     // [nfeatures,nitems]
               const float* lpar,  // [nfeatures,nclus,nlevels]
                     float* lprob) // [nclus,nitems]
{
  if(blocksize != blockDim.x) asm("trap;");

  int
    tid  = threadIdx.x,
    clus = blockIdx.x * blocksize + tid,
    item = blockIdx.y * blocksize + tid;

  x     += item;
  lpar  += clus * nlevels;
  lprob += item;

  float sum[blocksize] = { 0 };
  for (int feat = 0; feat < nfeatures; feat++) {
    extern __shared__ float lparblock[]; // [nlevels,clus=blocksize]

    // load the parameter values for our thread's cluster into shared memory
    if (clus < nclus) {
      for (int lev = 0; lev < nlevels; lev++) {
          lparblock[tid * nlevels + lev] = lpar[(feat * nclus) * nlevels + lev];
      }
    }

    // ensure all of this block's threads have made it up to here
    __syncthreads();

    if (item < nitems) {
      // get our thread items's feature value f
      const int xf = x[feat * nitems];
      // so it can be tested over all i cluster values
      for (int i = 0; i < blocksize; i++)
        sum[i] += lparblock[i * nlevels + xf];
    }
  }

  if (item < nitems) {
    for (int i = 0; i < blocksize; i++) {
      clus = blockIdx.x * blocksize + i;
      if (clus < nclus)
        lprob[clus * nitems] = sum[i];
    }
  }
}

void
cuda::multinomial::sampleParameters()
{
  const int
    blocksize = 16,
    blockclus = (gpu->nclus()+blocksize-1)/blocksize,
    blockitems = (gpu->nitems()+blocksize-1)/blocksize,
    sharedlen = sizeof(float)*nlevels*blocksize;

  multinom_samplepars<<<dim3(blockclus,nfeatures),blocksize,sharedlen>>>
    (gpu->nitems(), gpu->nclus(), nfeatures, nlevels, gpu->myfile(), d_data,
     gpu->mygpuinverseallocs(), d_lpar, gpu->ittr());
}

void
cuda::multinomial::accumAllocProbs()
{
  const int
    blocksize = 16,
    blockclus = (gpu->nclus()+blocksize-1)/blocksize,
    blockitems = (gpu->nitems()+blocksize-1)/blocksize,
    sharedlen = sizeof(float)*nlevels*blocksize;

  multinom_accumprobs<blocksize><<<dim3(blockclus,blockitems),blocksize,sharedlen>>>
    (gpu->nitems(), gpu->nclus(), nfeatures, nlevels, d_data, d_lpar,
     gpu->mygpuprobs());
}

__global__ void
bagofwords_samplepars (int nitems, int nclus, int nfeatures, int file,
                 const int*   x,     // [nfeatures,nitems]
                 const int*   ialloc, // [nitems+nclus+1]
                       float* lpar,  // [nfeatures,nclus]
                       int ittr)
{
  const int
    clus = blockIdx.x * blockDim.x + threadIdx.x,
    feat = blockIdx.y;

  if (clus < nclus) {
    x    += feat * nitems;
    lpar += feat * nclus + clus;

    int sum = 0;
    for (const int
	   *cur = ialloc + ialloc[clus],
	   *end = ialloc + ialloc[clus+1];
	 cur < end;
	 cur++) {
      sum += x[*cur];
    }

    // init RNG
    r123::distr rng((r123::distr::ukey_type){{ittr,clus * nfeatures + feat}});
    *lpar = rng.rgamma(0.5 + sum);
  }
}

__global__ void
bagofwords_normalise (int nclus, int nfeatures,
		      float* lpar) // [nfeatures,nclus]
{
  const int
    clus = blockIdx.x * blockDim.x + threadIdx.x;
  if (clus < nclus) {
    lpar += clus;
    float sum = 0;
    for (int feat = 0; feat < nfeatures; feat++) {
      sum += lpar[feat * nclus];
    }
    for (int feat = 0; feat < nfeatures; feat++) {
      lpar[feat * nclus] = __logf(lpar[feat * nclus] / sum);
    }
  }
}

template<int blocksize>
__global__ void
bagofwords_accumprobs (int nitems, int nclus, int nfeatures,
                 const int*   x,     // [nfeatures,nitems]
                 const float* lpar,  // [nfeatures,nclus]
                       float* lprob) // [nclus,nitems]
{
  if(blocksize != blockDim.x) asm("trap;");

  int
    tid  = threadIdx.x,
    clus = blockIdx.x * blocksize + tid,
    item = blockIdx.y * blocksize + tid;

  x     += item;
  lpar  += clus;
  lprob += item;

  float sum[blocksize] = { 0 };
  for (int feat = 0; feat < nfeatures; feat++) {
    __shared__ float bwblock[blocksize];

    // load the parameter values for our thread's cluster into shared memory
    if (clus < nclus)
      bwblock[tid] = lpar[feat * nclus];

    // ensure all of this block's threads have made it up to here
    __syncthreads();

    if (item < nitems) {
      // get our thread items's feature value
      const int xf = x[feat * nitems];
      // so it can be tested over all i cluster values
      for (int i = 0; i < blocksize; i++) {
       sum[i] += bwblock[i] * xf;
      }
    }
  }

  if (item < nitems) {
    if (0 && clus == 0) {
      printf("*** P(%i,%i..%i) = [%g,%g,%g,...]\n",
            item, blockIdx.x * blocksize, (blockIdx.x+1) * blocksize-1,
            sum[0], sum[1], sum[2]);
    }
    for (int i = 0; i < blocksize; i++) {
      clus = blockIdx.x * blocksize + i;
      if (clus < nclus)
        lprob[clus * nitems] = sum[i];
    }
  }
}

void
cuda::bagofwords::sampleParameters()
{
  const int
    blocksize  = 16,
    blockclus  = (gpu->nclus()+blocksize-1)/blocksize,
    blockitems = (gpu->nitems()+blocksize-1)/blocksize;

  bagofwords_samplepars<<<dim3(blockclus,nfeatures),blocksize>>>
    (gpu->nitems(), gpu->nclus(), nfeatures, gpu->myfile(), d_data,
     gpu->mygpuinverseallocs(), d_lpar, gpu->ittr());

  bagofwords_normalise<<<blockclus,blocksize>>>
    (gpu->nclus(), nfeatures, d_lpar);
}

void
cuda::bagofwords::accumAllocProbs()
{
  const int
    blocksize  = 16,
    blockclus  = (gpu->nclus()+blocksize-1)/blocksize,
    blockitems = (gpu->nitems()+blocksize-1)/blocksize;

  bagofwords_accumprobs<blocksize><<<dim3(blockclus,blockitems),blocksize>>>
    (gpu->nitems(), gpu->nclus(), nfeatures, d_data, d_lpar,
     gpu->mygpuprobs());
}

__device__ int
uppertri_idx(const int M, const int i, const int j)
{
  if (i == j) {
    asm("trap;");
  }
  const int d = j-i;
  if (d > 0)
    return M*i - (i*(i+1))/2 + d - 1;
  else
    return M*j - (j*(j+1))/2 - d - 1;
}

/** Apply any weighting (both within- and across-datasets) to
 * item/cluster probabilities and get them out of log-space and into a
 * cumulative sum.  Rescaling within items is needed to ensure that
 * probabilities will be within the range of a float.
 *
 * The probabilities $\hat\pi$ associated with item $i$ and cluster
 * $j$ are thus:
 * \begin{equation}
 * \hat\pi_{i,j} \propto \pi_j \, f\!\big(x_i \mid \theta_j\big)
 * \end{equation}
 * where $\hat\pi$ is scaled to form a cumlative sum within each item,
 * ranging from $0$ to $1$.
 */
__global__ void
logprob_to_probsums (int nfiles, int nitems, int nclus,
                     float* prob,    // [nfiles,nclus,nitems]
               const float* lweight, // [nclus,nfiles]
               const int*   alloc,   // [nfiles,nitems]
               const float* lrho)    // [nfiles*(nfiles-1)/2]
{
  const int
    item = blockIdx.x * blockDim.x + threadIdx.x,
    file = blockIdx.y;

  extern __shared__   float slrho[]; // [nfiles]

  if (threadIdx.x < nfiles) {
    float x = -INFINITY;
    if (threadIdx.x != file) {
      x = lrho[uppertri_idx(nfiles, file, threadIdx.x)];
    }
    slrho[threadIdx.x] = x;
  }
  __syncthreads();

  if (item < nitems) {
    prob    += file * nclus * nitems + item;
    lweight += file * nclus;
    alloc   += item;

    if (0) {
      printf("*** P(%3i, pre) = "
	     "[%8g,%8g,%8g,%8g,"
	     " %8g,%8g,%8g,%8g]\n",
	     item,
	     prob[nitems * 0], prob[nitems * 1], prob[nitems * 2], prob[nitems * 3],
	     prob[nitems * 4], prob[nitems * 5], prob[nitems * 6], prob[nitems * 7]);
    }

    float max = -INFINITY;
    for (int clus = 0; clus < nclus; clus += 4) {
      float
	w1 = lweight[clus],
	w2 = lweight[clus+1],
	w3 = lweight[clus+2],
	w4 = lweight[clus+3];
      for (int f = 0; f < nfiles; f++) {
	if (f == file)
	  continue;
        if (alloc[f * nitems] == clus)   w1 += slrho[f];
        if (alloc[f * nitems] == clus+1) w2 += slrho[f];
        if (alloc[f * nitems] == clus+2) w3 += slrho[f];
        if (alloc[f * nitems] == clus+3) w4 += slrho[f];
      }
      w1 = (prob[clus     * nitems] += w1);
      w2 = (prob[(clus+1) * nitems] += w2);
      w3 = (prob[(clus+2) * nitems] += w3);
      w4 = (prob[(clus+3) * nitems] += w4);
      w1 = fmaxf(w1,w2);
      w3 = fmaxf(w3,w4);
      max = fmaxf(max, fmaxf(w1,w3));
    }
    float sum = 0;
    for (int clus = 0; clus < nclus; clus += 4) {
      float
	p1 = prob[clus     * nitems],
	p2 = prob[(clus+1) * nitems],
	p3 = prob[(clus+2) * nitems],
	p4 = prob[(clus+3) * nitems];
      p1 = sum += __expf(p1 - max);
      p2 = sum += __expf(p2 - max);
      p3 = sum += __expf(p3 - max);
      p4 = sum += __expf(p4 - max);
      prob[clus     * nitems] = p1;
      prob[(clus+1) * nitems] = p2;
      prob[(clus+2) * nitems] = p3;
      prob[(clus+3) * nitems] = p4;
    }
    if (0) {
      printf("*** P(%3i,post) = "
	     "[%5.2f,%5.2f,%5.2f,%5.2f,"
	     " %5.2f,%5.2f,%5.2f,%5.2f]\n",
	     item,
	     prob[nitems * 0], prob[nitems * 1], prob[nitems * 2], prob[nitems * 3],
	     prob[nitems * 4], prob[nitems * 5], prob[nitems * 6], prob[nitems * 7]);
    }
  }
}

/** \function{samplealloc\_probsums} Perform a sampling of cluster
 * allocations given a set (unnormalised) probabilities of assignments
 * of items to each cluster.  A binary search is employed, though I'm
 * not sure whether this better than a linear search in \textsc{cuda}.
 * I \emph{think} it should be, given that we're expecting the number
 * of clusters (and hence the number of items we're searching over) to
 * be somewhere around 40 and 200---but I've not spent the time
 * benchmarking it.
 */
__global__ void
samplealloc_probsums (int nfiles, int nitems, int nclus,
                const float* prob,  // [nfiles,nclus,nitems]
                      int*   alloc, // [nfiles,nitems]
                      int    ittr)
{
  const int
    item = blockIdx.x * blockDim.x + threadIdx.x,
    file = blockIdx.y;
  if (item < nitems) {
    prob  += file * nclus * nitems + item;
    alloc += file * nitems + item;

    r123::distr rng((r123::distr::ukey_type){{ittr,item * nfiles + file}});
    const float p = rng.runif() * prob[(nclus-1) * nitems];

    // binary search for p in this item's list
    int begin = 0, end = nclus;
    while (begin < end) {
      // don't change to (begin+end)/2, this is modular integer arithmetic
      const int mid = begin + (end-begin)/2;
      if (prob[mid * nitems] < p)
        begin = mid+1;
      else
        end = mid;
    }
    *alloc = begin;
    if (0) {
      printf("*** (%3i,%3i) = %i [%5.2f <%5.2f <=%5.2f] "
	     "[%5.2f,%5.2f,%5.2f,%5.2f,"
	     " %5.2f,%5.2f,%5.2f,%5.2f]\n", item, file, begin,
	     begin > 0 ? prob[(begin-1) * nitems] : 0,
	     p,
	     prob[begin * nitems],

	     prob[0 * nitems], prob[1 * nitems], prob[2 * nitems], prob[3 * nitems],
	     prob[4 * nitems], prob[5 * nitems], prob[6 * nitems], prob[7 * nitems]);
    }
  }
}

void
cuda::sampler::sampleAlloc()
{
  assert(nfiles() == _nsamplers);

  logprob_to_probsums<<<dim3((nitems()+15)/16,nfiles()),16,sizeof(float)*nfiles()>>>
    (nfiles(), nitems(), nclus(), d_prob, d_lweight, d_alloc, d_lrho);

  samplealloc_probsums<<<dim3((nitems()+7)/8,nfiles()), 8>>>
    (nfiles(), nitems(), nclus(), d_prob, d_alloc, _ittr++);

  // the allocations have now been resampled, update our cache of "inverse allocations"
  allocChanged(d_alloc);
}
