#include "csv.hpp"

#include "datatypes.hpp"
#include "utility.hpp"

static normgam<> prior(0, 1, 2, 2);

gaussianSampler::gaussianSampler(class cuda::sampler *gpu,
				 const Eigen::MatrixXf &data, int nclus)
  : _nclus(nclus), _data(data),
    _mu(data.rows(), nclus+1), _tau(data.rows(), nclus+1),
    _featurestate(Eigen::VectorXi::Ones(data.rows()))
{
  if(gpu) {
#ifndef NOCUDA
    _gpu.gpu = gpu->nextFileSampler();
    _gpu.nfeatures = nfeatures();

    _gpu.d_data.resize(nitems() * nfeatures());
    _gpu.d_par.resize(nclus * nfeatures());

    _gpu.d_data = eigenMatrixToStdVector(data.transpose());
#endif
  } else {
    _gpu.gpu = 0;
    _gpu.nfeatures = 0;
  }
}

std::vector<runningstats<> >
gaussianSampler::accumState(const Eigen::Ref<const Eigen::VectorXi> &alloc)
{
  std::vector<runningstats<> > accum(nclus() * nfeatures());

  for (int i = 0; i < nitems(); i++) {
    int
      ii = alloc(i) * nfeatures();
    for (int f = 0; f < nfeatures(); f++) {
      accum[ii + f].push(_data(f, i));
    }
  }
  return accum;
}

void
gaussianSampler::sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc)
{
  std::vector<runningstats<> > accum(accumState(alloc));

  for (int j = 0; j < nclus(); j++) {
    for (int f = 0; f < nfeatures(); f++) {
      auto x = prior.update(accum[j * nfeatures() + f]).sample(generator);

      _mu(f,j)  = x.first;
      _tau(f,j) = x.second;
    }
  }
}

Eigen::ArrayXf
gaussianSampler::logprobItemClus(int item, int clus) const
{
  return 0.5*_tau.col(clus).array().log() - (_data.col(item)-_mu.col(clus)).array().square() * 0.5*_tau.col(clus).array();
}

Eigen::VectorXf
gaussianSampler::Item::operator()(int item) const
{
  Eigen::VectorXf out(s->nclus());
  for (int j = 0; j < s->nclus(); j++) {
    out[j] = (s->logprobItemClus(item, j) * s->_featurestate.array().cast<float>()).sum();
  }
  return out;
}

void
gaussianSampler::sampleFeatureSelection(const Eigen::Ref<const Eigen::VectorXi> &alloc)
{
  std::vector<runningstats<> > accum(accumState(alloc));

  // and sample for feature selection
  for (int f = 0; f < nfeatures(); f++) {
    runningstats<> rs;
    for (int j = 0; j < nclus(); j++) {
      rs += accum[j * nfeatures() + f];
    }
    auto x = prior.update(rs).sample(generator);
    _mu(f,nclus())  = x.first;
    _tau(f,nclus()) = x.second;
  }

  Eigen::VectorXf prob = Eigen::VectorXf(nfeatures());
  Eigen::VectorXi used = Eigen::VectorXi(nclus());
  // probability of data
  for (int i = 0; i < nitems(); i++) {
    int j = alloc(i);
    used[j] = 1;
    prob.array() += logprobItemClus(i, j);
    prob.array() -= logprobItemClus(i, nclus());
  }
  // prior probability
  for (int f = 0; f < nfeatures(); f++) {
    for (int j = 0; j < nclus(); j++) {
      if (used[j])
	prob(f) += prior.logpdf(_mu(f,j), _tau(f,j));
    }
    prob(f) -= prior.logpdf(_mu(f,nclus()), _tau(f,nclus()));
  }

  // move out of log-space and perform binomial draws
  prob = 1/(1+prob.array().exp());

  for (int f = 0; f < nfeatures(); f++) {
    _featurestate(f) = std::uniform_real_distribution<>()(generator) > prob(f);
  }
}
