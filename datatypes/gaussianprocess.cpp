#include "csv.hpp"

#include "datatypes.hpp"
#include "utility.hpp"

#include <sstream>

#define GPJITTER 0.001

static const Eigen::IOFormat CSVFMT(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n", "", "", "", "");

Eigen::MatrixXd
gp::covarianceMatrix (const Eigen::VectorXd &time) const
{
  size_t n = time.size();
  Eigen::MatrixXd mat = (((Eigen::MatrixXd::Zero(n,n).colwise() + time
			   ).rowwise() - time.transpose()
			  ).array().square() * (-0.5 * _pl)).exp() / _pf;
  if (_pn != 0)
    mat.diagonal().array() += 1/_pn;
  return mat;
}

void
gp::sampleFromPrior (const gpHypers &prior)
{
  _pf = prior.pf()(generator);
  _pn = prior.pn()(generator);
  _pl = prior.pl()(generator);
}

Eigen::VectorXd
gp::sampleFunction (const Eigen::VectorXd &time)
{
  return multivariatenormal_distribution<>(covarianceMatrix(time))(generator);
}

gpSampler::gpSampler(class cuda::sampler *gpu,
		     const Eigen::MatrixXf &data, const Eigen::VectorXd &time, int nclus)
  : _nclus(nclus), _data(data), _time(time),
    _sampfn(time.size(), nclus) {
  _gp.reserve(nclus);
  for (int i = 0; i < nclus; i++)
    _gp.emplace_back(_prior);

#ifndef NOCUDA
  if (gpu) {
    _gpu.gpu = gpu->nextFileSampler();
    _gpu.nfeatures = nfeatures();

    _gpu.d_data.resize(nfeatures() * nitems());
    _gpu.d_data = eigenMatrixToStdVector(data.transpose());

    _gpu.d_sampfn.resize(nfeatures() * nclus);
    _gpu.d_sigma.resize(nclus);
  } else {
    _gpu.gpu = 0;
    _gpu.nfeatures = 0;
  }
#endif
}

void
gpSampler::swap(const std::vector<int> &changed)
{
  std::vector<gp> arr;
  for (size_t i = 0; i < _gp.size(); i++)
    arr.emplace_back(_gp[changed[i]]);
  _gp = arr;

  // just need to swap the GP hyper parameters, the rest are Gibbs
  // sampled so don't depend on previous values
}

void
gpSampler::sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc)
{
  Eigen::VectorXd nuse = Eigen::VectorXd::Zero(nclus());
  Eigen::MatrixXd ysums = Eigen::MatrixXd::Zero(nfeatures(), nclus());

  for (int i = 0; i < nitems(); i++) {
    int ii = alloc(i);
    nuse(ii) += 1;
    ysums.col(ii) += _data.col(i).cast<double>();
  }

  for (int i = 0; i < nclus(); i++) {
    gp &gpi = _gp[i];
    if (nuse(i) == 0) {
      gpi.sampleFromPrior(_prior);
      _sampfn.col(i) = gpi.sampleFunction(_time).cast<float>();
    } else {
      // note we're working with precisions, not variances here

      // invNoiseMatrix = (1/sn2)*eye(nFeatures);
      Eigen::MatrixXd invnoise = Eigen::MatrixXd::Zero(_time.size(),_time.size());
      invnoise.diagonal() = Eigen::VectorXd::Constant(_time.size(), gpi.pn());

      // Sigma_post = inv(inv(covMat) + (currentNGenes*invNoiseMatrix));
      Eigen::MatrixXd cov = gp(gpi.pf(), 1/GPJITTER, gpi.pl()).covarianceMatrix(_time).inverse();
      cov.diagonal().array() += nuse(i) * gpi.pn();
      cov = cov.inverse();

      // mu_post    = Sigma_post*invNoiseMatrix*ysum(i,:)';
      Eigen::VectorXd mu_post = cov * invnoise * ysums.col(i);

      // f   = mvnrnd(mu_post, Sigma_post);
      _sampfn.col(i) = (mu_post + multivariatenormal_distribution<>(cov)(generator)).cast<float>();

      {
	const double
	  pn = gpi.pn(),
	  prop_pn = std::abs(pn + gaussian_distribution<>(5)(generator));

	double
	  ll1 = _prior.pn().logpdf(pn),
	  ll2 = _prior.pn().logpdf(prop_pn);

	gaussian_distribution<> d1(1/sqrt(pn)), d2(1/sqrt(prop_pn));

	for (int j = 0; j < nitems(); j++) {
	  if (i != alloc(j))
	    continue;
	  ll1 += (_data.col(j) - _sampfn.col(i)).unaryExpr([d1] (float x) { return d1.logpdf(x); }).sum();
	  ll2 += (_data.col(j) - _sampfn.col(i)).unaryExpr([d2] (float x) { return d2.logpdf(x); }).sum();
	}

	if (ll1 - std::exponential_distribution<>()(generator) < ll2) {
	  gpi.pn(prop_pn);
	}
      }

      {
	const double pf = gpi.pf(), prop_pf = std::abs(pf + gaussian_distribution<>(5)(generator));

	cov = gp(pf, 1/GPJITTER, gpi.pl()).covarianceMatrix(_time);
	double ll1 = _prior.pf().logpdf(pf) +
	  multivariatenormal_distribution<>(cov).logpdf(_sampfn.col(i).cast<double>());

	cov = gp(prop_pf, 1/GPJITTER, gpi.pl()).covarianceMatrix(_time);
	double ll2 = _prior.pf().logpdf(prop_pf) +
	  multivariatenormal_distribution<>(cov).logpdf(_sampfn.col(i).cast<double>());

	if (ll1 - std::exponential_distribution<>()(generator) < ll2) {
	  gpi.pf(prop_pf);
	}
      }

      {
	const double pl = gpi.pl(), prop_pl = std::abs(pl + gaussian_distribution<>(1)(generator));

	cov = gp(gpi.pf(), 1/GPJITTER, pl).covarianceMatrix(_time);
	double ll1 = _prior.pl().logpdf(pl) +
	  multivariatenormal_distribution<>(cov).logpdf(_sampfn.col(i).cast<double>());

	cov = gp(gpi.pf(), 1/GPJITTER, prop_pl).covarianceMatrix(_time);
	double ll2 = _prior.pl().logpdf(prop_pl) +
	  multivariatenormal_distribution<>(cov).logpdf(_sampfn.col(i).cast<double>());

	if (ll1 - std::exponential_distribution<>()(generator) < ll2) {
	  gpi.pl(prop_pl);
	}
      }

      //        std::cout << i+1 << ": " << gpi.pf() << ' ' << gpi.pn() << ' ' << gpi.pl() << '\n';
    }
  }
}

gpSampler::Item::Item(const gpSampler* s)
  : s(s), prob0(s->nclus()), prob1(s->nclus())
{
  for (int i = 0; i < s->nclus(); i++)
    prob1(i) = s->_gp[i].pn();

  prob0 = prob1.log() * (0.5*s->_time.size());
  prob1 *= 0.5;
}

Eigen::VectorXf
gpSampler::Item::operator()(int item) const
{
  /*
  for (int i = 0; i < nitems(); i++) {
    for (int j = 0; j < _nclus; j++) {
      gaussian_distribution<> d(1/sqrt(_gp[j].pn()));
      prob(j) = lweight(j) + (_data.col(gi) - _sampfn.col(j)).unaryExpr([d] (float x) {
	  return d.logpdf(x); }).sum();
    }
  }
  */
  return prob0 - (s->_sampfn.colwise() - s->_data.col(item)).array().square().colwise().sum().transpose().array() * prob1;
}

#ifndef NOCUDA
void gpSampler::cudaSampleParameters(const Eigen::Ref<const Eigen::VectorXi> &alloc)
{
  assert(_gpu.gpu != 0);

  // pass this off to the CPU code
  sampleParams (alloc);

  std::vector<float>
    sampfn = eigenMatrixToStdVector(_sampfn),
    sigma(nclus());
  for (int i = 0; i < nclus(); i++)
    sigma[i] = _gp[i].pn();

  // and move sampled parameters to the GPU
  _gpu.d_sigma  = sigma;
  _gpu.d_sampfn = sampfn;
}
#endif

gpDatatype::gpDatatype(const char *path)
 : csvdatastore(path), _time(_colnames.size())
{
  // convert column names into doubles to use as time points
  for (size_t i = 0; i < _colnames.size(); i++) {
    _time(i) = stod_strict(_colnames[i]);
  }
}
