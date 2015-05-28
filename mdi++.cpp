#include <fstream>
#include <iostream>
#include <algorithm>

#include <Eigen/Dense>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;

#include "csv.hpp"
#include "datatypes.hpp"
#include "interdataset.hpp"
#include "stats.hpp"
#include "cuda/cudasampler.hpp"
#include "utility.hpp"

#include <sys/time.h>

std::default_random_engine generator;

static const Eigen::IOFormat CSVFMT(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n", "", "", "", "");

class csvwriter {
  bool _firstcol, _colbreak;
  std::ostream &_out;

  void nextvalue() {
    if(_firstcol) {
      assert(!_colbreak);
      _firstcol = false;
      return;
    }
    _out << ',';
    if(_colbreak) {
      _out << ' ';
      _colbreak = false;
    }
  }

  template<typename T>
  void writeraw(const T&x) {
    nextvalue();
    _out << x;
  }

public:
  csvwriter(std::ostream &out) : _firstcol(true), _colbreak(false), _out(out) {}

  void endrow() {
    // make sure we don't output an empty row
    assert(!_firstcol);
    _out << std::endl;
    _firstcol = true;
    _colbreak = false;
  }

  void columnbreak() {
    if (!_firstcol)
      _colbreak = true;
  }

  void write(short i) { writeraw(i); }
  void write(int   i) { writeraw(i); }
  void write(long  i) { writeraw(i); }

  void write(float  f) { writeraw(f); }
  void write(double f) { writeraw(f); }

  void write(const std::string &s) {
    nextvalue();

    _out << '"';
    for (std::string::const_iterator i = s.begin(), end = s.end(); i != end; ++i) {
      unsigned char c = *i;
      if (' ' <= c && c <= '~' && c != '\\' && c != '"') {
	_out << c;
      } else {
	_out << '\\';
	switch(c) {
	case '"':  _out << '"';  break;
	case '\\': _out << '\\'; break;
	case '\t': _out << 't';  break;
	case '\r': _out << 'r';  break;
	case '\n': _out << 'n';  break;
	default:
	  char const* const hexdig = "0123456789ABCDEF";
	  _out << 'x';
	  _out << hexdig[c >> 4];
	  _out << hexdig[c & 0xF];
	}
      }
    }
    _out << '"';
  }
};

void
printHeader(const shared& shared,
            const std::vector<std::string> &ss,
            csvwriter &out)
{
  std::stringstream val;
  for (int k = 0; k < shared.nfiles(); k++) {
    val.str("");
    val << "MassParameter_" << k+1;
    out.write(val.str());
  }
  out.columnbreak();
  for (int k = 0; k < shared.nfiles()-1; k++) {
    for (int l = k+1; l < shared.nfiles(); l++) {
      val.str("");
      val << "Phi_" << k+1 << l+1;
      out.write(val.str());
    }
  }
  out.columnbreak();
  for (int k = 0, a = 0; k < shared.nfiles(); k++) {
    for (int i = 0; i < shared.nitems(); i++) {
      val.str("");
      val << "Dataset" << k+1 << '_' << ss[a++];
      out.write(val.str());
    }
  }
  out.endrow();
}

void
printSample(const shared &shared, const mdisampler &mdi,
	    csvwriter &out)
{
  for (int k = 0; k < shared.nfiles(); k++) {
    out.write(mdi.dpmass(k));
  }
  out.columnbreak();
  for (int k = 0; k < shared.nfiles()-1; k++) {
    for (int l = k+1; l < shared.nfiles(); l++) {
      out.write(mdi.phi(k, l));
    }
  }
  for (int k = 0; k < shared.nfiles(); k++) {
    out.columnbreak();
    for (int i = 0; i < shared.nitems(); i++) {
      out.write(shared.alloc(k, i)+1);
    }
  }
  out.endrow();
}

void
printFeatureHeader(const std::vector<std::string> &ss,
                   csvwriter &out)
{
  for (size_t i = 0; i < ss.size(); i++) {
    out.write(ss[i]);
  }
  out.endrow();
}

void
printFeatureSample(const std::vector<std::unique_ptr<sampler> > &ss,
                   csvwriter &out)
{
  for (const auto &s : ss) {
    auto fs = s->featureState();
    out.columnbreak();
    for (int i = 0; i < fs.size(); i++) {
      out.write(fs[i]);
    }
  }
  out.endrow();
}

static gamma_distribution<> phiprior(1.0, 0.2), dpprior(2, 4);

double
sampleMdiNu (const mdisampler &mdi, const shared &shared)
{
  auto d = gamma_distribution<>(shared.nitems(), mdi.nuConditionalBeta());
  return d(generator);
}

double
sampleMdiPhi (const mdisampler &mdi, const shared &shared, int k, int l)
{
  auto d = gamma_distribution<>(phiprior.shape() + shared.phiConditionalAlpha(k, l),
				phiprior.rate() + mdi.phiConditionalBeta(k, l));
  return d(generator);
}

Eigen::MatrixXf
sampleMdiWeights(const mdisampler &mdi, const shared &shared)
{
  Eigen::MatrixXi clussize(Eigen::MatrixXi::Zero(shared.nfiles(), shared.nclus()));
  for (int i = 0; i < shared.nitems(); i++) {
    for (int k = 0; k < shared.nfiles(); k++) {
      clussize(k, shared.alloc(k, i)) += 1;
    }
  }

  Eigen::MatrixXf weight(mdi.nclus(),mdi.nfiles());

  // sample weights
  for (int k = 0; k < mdi.nfiles(); k++) {
    const gamma_distribution<> prior(mdi.dpmass(k) / mdi.nclus(), 1.0);
    for (int j = 0; j < mdi.nclus(); j++) {
      auto d = gamma_distribution<>(prior.shape() + shared.gammaConditionalAlpha(k,j),
				    prior.rate() + mdi.gammaConditionalBeta(k, j));
      weight(j,k) = d(generator);
    }
  }

  return weight;
}

Eigen::VectorXi
sampleSharedAlloc(const mdisampler &mdi, const shared &shared, int k, const sampler::item &is)
{
  Eigen::VectorXi alloc(shared.nitems());
  Eigen::VectorXf
    prob(shared.nclus()),
    lweight(mdi.weight(k).array().log().cast<float>()),
    lrho((mdi.phis(k).array().unaryExpr(std::ptr_fun(log1p)).cast<float>()));

  for (int i = 0; i < shared.nitems(); i++) {
    prob = lweight + is(i);

    for (int l = 0; l < shared.nfiles(); l++) {
      if (k == l) continue;
      prob(shared.alloc(l, i)) += lrho(l);
    }

    prob = (prob.array() - prob.maxCoeff()).exp();

    alloc(i) = std::discrete_distribution<>(prob.data(), prob.data() + prob.size()-1)(generator);
  }

  return alloc;
}

double
sampleDpMass (const mdisampler &mdi, int k)
{
  const double
    dp1 = mdi.dpmass(k),
    dp2 = std::abs(std::normal_distribution<>(dp1, 1)(generator));

  double ll1 = dpprior.logpdf(dp1), ll2 = dpprior.logpdf(dp2);

  const gamma_distribution<>
    d1(dp1 / mdi.nclus(), 1.0),
    d2(dp2 / mdi.nclus(), 1.0);
  for (int j = 0; j < mdi.nclus(); j++) {
    ll1 += d1.logpdf(mdi.weight(j,k));
    ll2 += d2.logpdf(mdi.weight(j,k));
  }

  return mcmcAcceptProposalLogProb(ll1, ll2) ?
    dp2 : dp1;
}

std::vector<int>
sampleAllocBlockSwap (const shared &shared, const mdisampler &mdi, int k)
{
  const Eigen::VectorXd lrho = mdi.phis(k).array().unaryExpr(std::ptr_fun(log1p));

  Eigen::VectorXi alloc  = shared.alloc(k);
  Eigen::VectorXf weight = mdi.weight(k);

  std::vector<int> changed(shared.nclus());
  for (int i = 0; i < shared.nclus(); i++) {
    changed[i] = i;
  }

  for (int c1 = 0; c1 < shared.nclus(); c1++) {
    int c2 = std::uniform_int_distribution<int>(0, (int)shared.nclus()-2)(generator);
    c2 += c2 >= c1;

    // for keeping track of the accept/reject ratio
    double llratio = 0;

    for (int i = 0; i < alloc.size(); i++) {
      if (alloc(i) == c1 || alloc(i) == c2) {
        for (int l = 0; l < shared.nfiles(); l++) {
          if (k == l) continue;
          if (shared.alloc(l,i) == c1) {
            llratio += alloc(i) == c2 ? -lrho(l) : lrho(l);
          }
          if (shared.alloc(l,i) == c2) {
            llratio += alloc(i) == c1 ? -lrho(l) : lrho(l);
          }
        }
      }
    }

    if (mcmcAcceptProposalLogProb(llratio)) {
      for (int i = 0; i < alloc.size(); i++) {
        if (alloc(i) == c1)
          alloc(i) = c2;
        else if(alloc(i) == c2) {
          alloc(i) = c1;
        }
      }
      std::swap(weight(c1),  weight(c2));
      std::swap(changed[c1], changed[c2]);
    }
  }

  return changed;
}

int
main(int argc, char **argv)
{
  // small pause to let the debugger attach
  usleep(10*1000);

  int nsamples, thinby, nclusters, nitems = -1;
  po::options_description genopt("General Options");
  genopt.add_options()
    ("help,h",                                                     "Display this help message")
    ("samples,n", po::value<int>(&nsamples)->default_value(10000), "Number of MCMC samples to generate")
    ("thin,t", po::value<int>(&thinby)->default_value(1),          "Thin output to one sample every 'arg' samples")
    ("clusters,c",po::value<int>(&nclusters)->default_value(50),   "Assume a maximum of 'arg' clusters")
    ("featuresel,f",po::value<std::string>(),                      "Perform feature selection")
#ifndef NOCUDA
    ("gpu,g",                                                      "Enable GPU support")
#endif
    ("benchmark",                                                  "Display wall-clock time taken to run computation")
    ("seed,s",po::value<unsigned>(),                               "Seed for Random Number Generation")
    ("version,v",                                                  "Display the version number")
    ;

  std::vector<std::string> inputs;
  po::options_description hiddenopt("Hidden options");
  hiddenopt.add_options()
    ("input-files", po::value<std::vector<std::string>>(&inputs), "datatypes and files")
    ;

  po::variables_map vm;
  try {
    po::options_description cmdopts;
    cmdopts.add(genopt).add(hiddenopt);
    po::positional_options_description p;
    p.add("input-files", -1);
    po::store(po::command_line_parser(argc, argv).options(cmdopts).positional(p).run(), vm);
    po::notify(vm);
  } catch (const std::logic_error &err) {
    std::cerr << "Error parsing command line: " << err.what() << '\n';
    return 1;
  }

  if(vm.count("help")) {
    po::options_description opts("MDI++ Command Line Help");
    opts.add(genopt);
    std::cout << opts << "\n"
      "  N  filename1 [filename2..n] : load files as independent Gaussian features\n"
      "  GP filename1 [filename2..n] : load files as Gaussian process data\n"
      "  M  filename1 [filename2..n] : load files as Multinomial features\n"
      "  BW filename1 [filename2..n] : load files as Bag-of-Words features \n";
    return 0;
  }

  if(vm.count("version")) {
    std::cout << "MDI++ version 0.2\n";
    return 0;
  }

  if (vm.count("seed")) {
    generator.seed(vm["seed"].as<unsigned>());
  } else {
    unsigned seed = (std::random_device())();
    std::cerr << "RNG Seed = " << seed << '\n';
    generator.seed(seed);
  }

  std::unique_ptr<std::ofstream> featurefile;
  std::unique_ptr<csvwriter> featuresel;
  if (vm.count("featuresel")) {
    featurefile.reset(new std::ofstream(vm["featuresel"].as<std::string>(), std::ofstream::out));
    featuresel.reset(new csvwriter(*featurefile));
  }

  std::vector<std::unique_ptr<datatype> > datasets;
  std::vector<std::string> itemnames, featurenames;

  enum {
    MODE_UNKNOWN,
    MODE_GAUSSIAN,
    MODE_GAUSSIANPROCESS,
    MODE_MULTINOMIAL,
    MODE_BAGOFWORDS
  } mode = MODE_UNKNOWN;
  for (auto const &name : inputs) {
    if (name == "N")  { mode = MODE_GAUSSIAN;        continue; }
    if (name == "GP") { mode = MODE_GAUSSIANPROCESS; continue; }
    if (name == "M")  { mode = MODE_MULTINOMIAL;     continue; }
    if (name == "BW") { mode = MODE_BAGOFWORDS;      continue; }

    std::unique_ptr<datatype> dst;
    switch(mode) {
    case MODE_UNKNOWN:
      std::cerr << "Error: Unknown datatype!\n";
      exit(1);
    case MODE_GAUSSIAN:        dst.reset(new gaussianDatatype  (name.c_str())); break;
    case MODE_GAUSSIANPROCESS: dst.reset(new gpDatatype        (name.c_str())); break;
    case MODE_MULTINOMIAL:     dst.reset(new multinomDatatype  (name.c_str())); break;
    case MODE_BAGOFWORDS:      dst.reset(new bagofwordsDatatype(name.c_str())); break;
    }
    int i = dst->items().size();
    if (datasets.size() == 0) {
      nitems = i;
    } else if (nitems != i) {
      std::cerr << "Error datafile " << name << " doesn't have the same number of items to cluster.\n";
      exit(1);
    }
    {
      auto n = dst->items();
      std::copy(n.begin(), n.end(), std::back_inserter(itemnames));
    }
    {
      auto n = dst->features();
      std::copy(n.begin(), n.end(), std::back_inserter(featurenames));
    }
    // finally(!) move here
    datasets.push_back(std::move(dst));
  }

  if (datasets.size() == 0) {
    std::cerr << "Error: Need to load at least one datafile.  Run with -h for help\n";
    exit(1);
  }

  const int nfiles = datasets.size();
  shared shared(nfiles, nclusters, nitems);
  const interdataset inter(nfiles);
  mdisampler mdi(inter, nclusters);

  shared.sampleFromPrior();
  mdi.sampleFromPrior();

  class cuda::sampler *gpu = 0;

  if (vm.count("gpu")) {
#ifdef NOCUDA
    std::cerr << "error: CUDA support not enabled in this build!";
    exit(1);
#else
    gpu = new cuda::sampler
      (nfiles, nitems, nclusters,
       inter.getPhiOrd(), inter.getWeightOrd());
    gpu->setIttr(generator());
    gpu->setNu(mdi.nu());
    gpu->setAlloc(eigenMatrixToStdVector(shared.allocs()));
    gpu->setPhis(eigenMatrixToStdVector(mdi.phis()));

    if (0) {
      std::cerr << mdi.weights() << "\n";
      std::vector<float> weights(eigenMatrixToStdVector(mdi.weights()));

      std::cerr << "weights: ";
      for (auto &x:weights) {
	std::cerr << x << ", ";
      }
      std::cerr << "\n";
    }

    gpu->setWeights(eigenMatrixToStdVector(mdi.weights()));
    gpu->setDpMass(eigenMatrixToStdVector(mdi.dpmass()));

#endif
  }

  std::vector<std::unique_ptr<sampler> > samplers;
  for (auto &dst : datasets) {
    samplers.emplace_back(dst->newSampler(nclusters, gpu));
    // TODO: add an explicit "sample from prior" call in here
  }

  csvwriter mcmcwriter(std::cout);
  printHeader (shared, itemnames, mcmcwriter);
  if(featuresel)
    printFeatureHeader(featurenames, *featuresel);

  // record the start time now initialisation is complete
  struct timeval starttime;
  gettimeofday (&starttime, 0);

  for (int ittr = 0; ittr < nsamples; ittr++) {
    if (gpu) {
#ifndef NOCUDA
      for (int k = 0; k < nfiles; k++) {
	auto &s = samplers[k];
	s->cudaSampleParameters(shared.alloc(k));
	s->cudaAccumAllocProbs();
      }

      gpu->sampleAlloc();
      gpu->sampleNuConditional();
      gpu->samplePhiConditionals();
      gpu->sampleGammaConditionals();
      gpu->sampleDpMass();

      shared.setAlloc(stdVectorToEigenMatrix(gpu->getAlloc(), nitems, nfiles));
      mdi.setWeight(stdVectorToEigenMatrix(gpu->getWeights(), nclusters, nfiles));
      mdi.setPhis(stdVectorToEigenVector(gpu->getPhis(), mdi.nphis()));
      mdi.setDpMass(stdVectorToEigenVector(gpu->getDpMass(), nfiles));
#endif
    } else {
      mdi.setNu(sampleMdiNu(mdi, shared));

      for (int k = 0; k < nfiles; k++) {
	mdi.setDpMass(k, sampleDpMass(mdi, k));
      }

      for (int k = 0; k < nfiles; k++) {
	auto &s = samplers[k];
	s->sampleParams(shared.alloc(k));

	std::unique_ptr<sampler::item> is(s->newItemSampler());
	const Eigen::VectorXi alloc(sampleSharedAlloc(mdi, shared, k, *is.get()));

	shared.setAlloc(k, alloc);
	if (featuresel)
	  s->sampleFeatureSelection(alloc);
      }

      // sample phi
      for (int k = 0; k < nfiles-1; k++) {
	for (int l = k+1; l < nfiles; l++) {
	  mdi.setPhi(k, l, sampleMdiPhi(mdi, shared, k, l));
	}
      }

      mdi.setWeight(sampleMdiWeights(mdi, shared));
    }

    // block update of cluster labels (e.g. swap all 1s and 3s in a given dataset)
    // the reason for doing this is to get labels to agree between datasets
    if (nfiles > 1) {
      for (int k = 0; k < nfiles; k++) {
	const std::vector<int> changed = sampleAllocBlockSwap (shared, mdi, k);

	std::vector<int> order(changed.size());
	for (size_t j = 0; j < order.size(); j++)
	  order[j] = j;

        std::sort(order.begin(), order.end(), [changed] (int i, int j) {
	    return changed[i] < changed[j]; });

        auto a1 = shared.alloc(k);
        Eigen::VectorXi a2(a1.size());
        for (int i = 0; i < a1.size(); i++) {
          a2(i) = order[a1(i)];
        }

        auto w1 = mdi.weight(k);
        Eigen::VectorXf w2(w1.size());
        for (int j = 0; j < w1.size(); j++) {
          w2(j) = w1(changed[j]);
        }

        shared.setAlloc(k, a2);
        mdi.setWeight(k, w2);
        samplers[k]->swap(changed);
      }

      if (gpu) {
#ifndef NOCUDA
	gpu->setAlloc(eigenMatrixToStdVector(shared.allocs()));
	gpu->setWeights(eigenMatrixToStdVector(mdi.weights()));
#endif
      }
    }

    // TODO: perform split merge steps, ala Savage et al. 2013 ICML

    // Output MCMC samples to the user, thinned as requested
    if (ittr % thinby == 0) {
      printSample(shared, mdi, mcmcwriter);
      if (featuresel)
        printFeatureSample(samplers, *featuresel);
    }
  }

  // print out benchmarking info
  if (vm.count("benchmark")) {
    struct timeval endtime;
    gettimeofday (&endtime, 0);

    std::cerr
      << nsamples << ','
      << (endtime.tv_sec - starttime.tv_sec) * 1000 + (endtime.tv_usec - starttime.tv_usec) / 1000 << " ms"
      << '\n';
  }

#ifndef NOCUDA
  if (gpu) {
    delete gpu;
  }
#endif

  return 0;
}
