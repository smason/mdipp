#include <fstream>
#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "datatypes.hpp"
#include "utility.hpp"

std::default_random_engine generator;

template<typename T>
void
writeCsvMatrix (std::ostream &fd,
		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data,
		const Eigen::VectorXi &alloc)
{
  fd << "cluster";
  for (int i = 0; i < data.cols(); i++) {
    fd << ",f" << i+1;
  }
  fd << "\n";
  for (int i = 0; i < data.rows(); i++) {
    fd << "c" << alloc(i)+1;
    for (int j = 0; j < data.cols(); j++) {
      fd << "," << data(i,j);
    }
    fd << "\n";
  }
}

template<typename T> void
checkEqualTemplate (T a, T b, const std::string &msg)
{
  if (a == b)
    return;
  std::cerr
    << "Error: " << msg
    << " (expecting " << a << " == " << b << ")"
    << std::endl;
}

void checkEqual(int a, int b, const std::string &msg) {
  return checkEqualTemplate(a, b, msg); }

template<typename T> void
checkApproxEqual(const T &a, const T &b,
		 const std::string &msg,
		 typename T::Scalar epsilon = 1e-6)
{
  // Eigen::Array<bool, T::RowsAtCompileTime, T::ColsAtCompileTime>
  // em = (a - b).array() <= a.array().abs().min(b.array().abs()) * epsilon;
  if (a.isApprox(b, epsilon))
    return;
  std::cerr
    << "Error: " << msg
    << " (matrix equality check failed at epsilon=" << epsilon << ")"
    << " differences observed: " << std::endl
    << (a-b).array().abs() << std::endl
    << std::endl;
}

void
checkApproxEqual(const double a, const double b,
		 const std::string &msg, double epsilon = 1e-6)
{
  using std::abs;
  using std::min;
  epsilon *= min(abs(a),abs(b));
  if(abs(a-b) <= epsilon)
    return;
  std::cerr
    << "Error: " << msg
    << " (expecting ||" << a << " - " << b << "|| <= " << epsilon << ")"
    << std::endl;
}


template<typename T> void
checkProbItemMassGt(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &prob,
		    int item, T mass, const std::string &msg)
{
  // normalise, nans and infinities will propagate!
  auto probnorm = prob.array() / prob.sum();

  // check all positive
  if ((probnorm < 0).any()) {
    std::cerr
      << "Error: " << msg
      << " (some probabilities are not positive)\n";
  }
  if (probnorm(item) < mass) {
    std::cerr
      << "Error: " << msg
      << " (mass of item " << item
      << " is " << probnorm(item)
      << ", i.e. < " << mass << ")\n";
  }
}

double
nuConditionalAlphaOracle(const Eigen::MatrixXi &alloc)
{
  return alloc.rows();
}

double
phiConditionalAlphaOracle(const Eigen::MatrixXi &alloc,
			  int m, int p)
{
  int sum = 0;
  for (int i = 0; i < alloc.rows(); i++) {
    sum += alloc(i,m) == alloc(i,p);
  }
  return 1 + sum;
}

double
gammaConditionalAlphaOracle(const Eigen::MatrixXi &alloc,
			  int m, int jm)
{
  int sum = 0;
  for (int i = 0; i < alloc.rows(); i++) {
    sum += alloc(i,m) == jm;
  }
  return 1 + sum;
}

/* the following set of functions work by having an @outer function
 * that works "down" through the files, setting $j_i$ to the
 * appropriate value and then calling the next @outer function.  Once
 * $j$ has been set for all $K$ files (i.e. @{i == K}), @outer defers
 * to @inner and the product for the given $j$ is evaluated.
 */

// See Equation 2 in section "A.2 Normalising Constant" of Kirk et.al 2012
double
nuConditionalBetaOracle(const mdisampler &mdi)
{
  const int N = mdi.nclus(), K = mdi.nfiles();

  std::vector<int> j(mdi.nfiles());

  auto inner = [K,&mdi,&j]() {
    long double prod = 1;
    for (int k = 0; k < K; k++) {
      prod *= mdi.weight(j[k], k);
    }
    for (int k = 0; k < K-1; k++) {
      for (int l = k+1; l < K; l++) {
	prod *= 1 + mdi.phi(k, l) * (j[k] == j[l]);
      }
    }
    return prod;
  };
  std::function<long double(int)> outer = [&outer, &inner, &j, N, K](int i) {
    if (i == K)
      return inner();

    long double sum = 0;
    for (j[i] = 0; j[i] < N; j[i]++)
      sum += outer(i+1);
    return sum;
  };
  return outer(0);
}

// See Kirk et.al 2012 $b_\phi$ from "Conditional for $\phi_{mp}$"
double
phiConditionalBetaOracle(const mdisampler &mdi,
			int m, int p)
{
  const int N = mdi.nclus(), K = mdi.nfiles();

  std::vector<int> j(mdi.nfiles());

  auto inner = [K,&mdi,&j,m,p]() {
    long double prod = 1;
    for (int k = 0; k < K; k++) {
      prod *= mdi.weight(j[k], k);
    }
    for (int k = 0; k < K-1; k++) {
      for (int l = k+1; l < K; l++) {
	if (l != p)
	  prod *= 1 + mdi.phi(k, l) * (j[k] == j[l]);
      }
    }
    for (int k = 0; k < p; k++) {
      if (k != m)
	prod *= 1 + mdi.phi(k, p) * (j[k] == j[p]);
    }
    return prod;
  };
  std::function<long double(int)> outer = [&outer,&inner,&j,N,K,m,p](int i) {
    if (i == K)
      return inner();

    // $j_m$ and $j_p$ are set outside this iteration
    if (i == m || i == p)
      return outer(i+1);

    long double sum = 0;
    for (j[i] = 0; j[i] < N; j[i]++)
      sum += outer(i+1);
    return sum;
  };
  // iterate over $\sum_{j_m = j_p = 1}^N$
  long double sum = 0;
  for (int i = 0; i < N; i++) {
    j[m] = j[p] = i;
    sum += outer(0);
  }
  return mdi.nu() * sum;
}

// See Kirk et.al 2012 $b_\gamma$ from "Conditional for $\gamma_{j_m m}$"
double
gammaConditionalBetaOracle (const mdisampler &mdi,
			   const int m, const int jm)
{
  const int N = mdi.nclus(), K = mdi.nfiles();

  std::vector<int> j(mdi.nfiles());
  // within data @m we are interested in cluster @jm
  j[m] = jm;

  auto inner = [K,m,&mdi,&j]() {
    long double prod = 1;
    for (int k = 0; k < K; k++) {
      if(k != m)
	prod *= mdi.weight(j[k], k);
    }
    for (int k = 0; k < K-1; k++) {
      for (int l = k+1; l < K; l++) {
	prod *= 1 + mdi.phi(k, l) * (j[k] == j[l]);
      }
    }
    return prod;
  };
  std::function<long double(int)> outer = [&outer, &inner, &j, N, K, m](int i) {
    if (i == K)
      return inner();

    if (i == m)
      return outer(i+1);

    long double sum = 0;
    for (j[i] = 0; j[i] < N; j[i]++)
      sum += outer(i+1);
    return sum;
  };
  return mdi.nu() * outer(0);
}

int
main()
{
  /* datatypes
   *
   * * loading data (CPU)
   *
   * * summarising data given cluster allocations (CPU GPU)
   *
   * * sampling cluster parameters given above summary (CPU GPU)
   *
   * * sampling cluster allocations given cluster parameters (CPU GPU)
   *
   * MDI prior
   *
   * * Sampling Nu given [weights and allocations]?
   *
   * * Sampling	weights given [lots!]
   *
   * * Sampling	DPs given weights and nu
   */

  generator.seed(1);

  // given a single "file" for each datatype, load it in define
  // allocations (same for all four files), weights, nu, DP
  // concentration

  const int
    nfiles = 4,
    nitems = 5,
    nclus = 10,
    ngaussfeatures = 3;

  // 5 items, cluster allocations [0 0 1 1 2]. one file for each
  // datatype
  Eigen::MatrixXi alloc(nitems, nfiles);
  for (int i = 0; i < nfiles; i++)
    alloc.col(i) << 0, 0, 1, 1, 2;
  Eigen::MatrixXf
    weights = alloc.cast<float>();
  Eigen::VectorXf
    dpmass = Eigen::VectorXf::Ones(nfiles);

  Eigen::MatrixXf data_gaussian(nitems, ngaussfeatures);
  data_gaussian <<
    -2.1, -2.1, -2.1,
    -1.9, -1.9, -1.9,
     1.9,  1.9,  1.9,
     2.1,  2.1,  2.1,
     5.0,  5.0,  5.0;

  // write dummy data out, load it back in and make sure it's the same
  // as the original data
  {
    std::ofstream out("data_gauss.txt");
    writeCsvMatrix (out, data_gaussian, alloc.col(0));
  }

  gaussianDatatype dt_gauss("data_gauss.txt");

  checkEqual(nitems, dt_gauss.items().size(),
	     "number of items read from gaussian dataset");
  checkEqual(ngaussfeatures, dt_gauss.features().size(),
	     "number of features read from gaussian dataset");

  checkApproxEqual<Eigen::MatrixXf>(data_gaussian.transpose(), dt_gauss.rawdata(),
				    "gaussian data from file");

  // given weights, allocations, nu:
  shared shared(nfiles, nclus, nitems);
  interdataset inter(nfiles);
  mdisampler mdi(inter, nclus);

  shared.sampleFromPrior();
  mdi.sampleFromPrior();

  shared.setAlloc(alloc);

  cuda::sampler	cuda(nfiles, nitems, nclus,
		     inter.getPhiOrd(), inter.getWeightOrd());
  cuda.setNu(mdi.nu());
  cuda.setDpMass(eigenMatrixToStdVector(mdi.dpmass()));
  cuda.setAlloc(eigenMatrixToStdVector(alloc));
  cuda.setPhis(eigenMatrixToStdVector(mdi.phis()));
  cuda.setWeights(eigenMatrixToStdVector(mdi.weights()));

  gaussianSampler * gauss_sampler = dt_gauss.newSampler(nclus, &cuda);

  // given known allocations, calculate Gaussian summaries
  {
    gauss_sampler->cudaSampleParameters(alloc.col(0));
    gauss_sampler->cudaAccumAllocProbs();

    const std::vector<runningstats<> > state(gauss_sampler->accumState(alloc.col(0)));
    checkEqual(nclus * ngaussfeatures, state.size(),
	       "number of accumulated gaussian stats");

    // check Gaussian summaries are OK
    const runningstats<> rs[nclus] = {{2,-2,0.02},{2,2,0.02},{1,5,0}};
    bool ok = true;
    for (int j = 0; j < nclus; j++) {
      for (int i = 0; i < ngaussfeatures; i++) {
	const runningstats<>
	  &a = state[j * ngaussfeatures + i],
	  &b = rs[j];
	ok &= a.isApprox(b);
      }
    }
    if (!ok) {
      std::cout << "Error: some of the accumulated gaussian stats are incorrect\n";
    }
  }

  // given known cluster parameters, ...
  {
    Eigen::MatrixXf mu(ngaussfeatures, nclus), tau(ngaussfeatures,nclus);
    mu.fill(0); tau.fill(20);
    mu.col(0).fill(-2);
    mu.col(1).fill( 2);
    mu.col(2).fill( 5);
    gauss_sampler->debug_setMuTau(mu, tau);
  }
  // ... sample Gaussian cluster association probabilities
  {
    std::unique_ptr<sampler::item> is(gauss_sampler->newItemSampler());
    for (int i = 0; i < nitems; i++) {
      Eigen::VectorXf prob((*is)(i));
      prob = (prob.array() - prob.maxCoeff()).exp();
      checkProbItemMassGt<float>(prob, alloc(i,0), 0.5,
				 "gaussian cluster allocations");
    }
  }

  {
    double
      oracle = nuConditionalBetaOracle(mdi);
    checkApproxEqual(oracle, mdi.nuConditionalBeta(),
		     "MDI nu beta conditional, CPU code");
    checkApproxEqual(oracle, cuda.collectNuConditionalBeta(),
		     "MDI nu beta conditional, GPU code");

    oracle = nuConditionalAlphaOracle(alloc);
    checkApproxEqual(oracle, shared.nuConditionalAlpha(),
		     "MDI nu alpha conditional, CPU code");
  }

  {
    Eigen::MatrixXf oracle, cpu, gpu;
    oracle = cpu = gpu = Eigen::MatrixXf::Zero(nfiles, nfiles);
    std::vector<float> gpuv = cuda.collectPhiConditionalsBeta();

    for (int k = 0, i = 0; k < nfiles; k++) {
      for (int l = k+1; l < nfiles; l++) {
	oracle(k,l) = phiConditionalBetaOracle(mdi, k, l);
	cpu(k,l)    = mdi.phiConditionalBeta(k, l);
	gpu(k,l)    = gpuv[i++];
      }
    }
    checkApproxEqual(oracle, cpu, "MDI phi beta conditionals, CPU code");
    checkApproxEqual(oracle, gpu, "MDI phi beta conditionals, GPU code");

    gpuv = cuda.collectPhiConditionalsAlpha();

    for (int k = 0, i = 0; k < nfiles; k++) {
      for (int l = k+1; l < nfiles; l++) {
	oracle(k,l) = phiConditionalAlphaOracle(alloc, k, l);
	cpu(k,l)    = shared.phiConditionalAlpha(k, l) + 1; // what to do about prior?
	gpu(k,l)    = gpuv[i++];
      }
    }
    checkApproxEqual(oracle, cpu, "MDI phi alpha conditionals, CPU code");
    checkApproxEqual(oracle, gpu, "MDI phi alpha conditionals, GPU code");
  }

  {
    Eigen::MatrixXf oracle, cpu, gpu;
    oracle = cpu = Eigen::MatrixXf::Zero(nclus, nfiles);

    gpu = stdVectorToEigenMatrix(cuda.collectGammaConditionalsBeta(),
				 nclus, nfiles);

    // take out the prior, not sure what to do about this.  I think I
    // should be checking priors as well, but conflating it in the
    // test unnecessarily seems to make things more difficult to debug
    // than it could...  hum!
    gpu.array() -= 1;

    for (int k = 0; k < nfiles; k++) {
      for (int j = 0; j < nclus; j++) {
	oracle(j,k) = gammaConditionalBetaOracle(mdi, k, j);
	cpu(j,k)    = mdi.gammaConditionalBeta(k, j);
      }
    }

    checkApproxEqual(oracle, cpu, "MDI gamma beta conditionals, CPU code");
    checkApproxEqual(oracle, gpu, "MDI gamma beta conditionals, GPU code");
  }

  return 0;
}
