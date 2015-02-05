#include "csv.hpp"

#include "datatypes.hpp"
#include "utility.hpp"

bagofwordsSampler::bagofwordsSampler(class cuda::sampler *gpu, const Eigen::MatrixXi &data, int nclus)
  : _nclus(nclus), _data(data),
    _lpar(data.rows(), nclus)
{
  _lpar.setZero();

  if(gpu) {
#ifndef NOCUDA
    _gpu.gpu = gpu->nextFileSampler();
    _gpu.nfeatures = nfeatures();

    _gpu.d_data.resize(nitems() * nfeatures());
    _gpu.d_lpar.resize(nclus * nfeatures());

    _gpu.d_data = eigenMatrixToStdVector(data.transpose());
#endif
  } else {
    _gpu.gpu = 0;
    _gpu.nfeatures = 0;
  }
}

void
bagofwordsSampler::sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc)
{
  _lpar.setConstant(0.5);
  for (int item = 0; item < nitems(); item++) {
    _lpar.col(alloc[item]) += _data.col(item).cast<float>().array();
  }

  _lpar = _lpar.array().unaryExpr([] (float alpha) {
      return gamma_distribution<float>(alpha, 1)(generator);
    });

  _lpar = (_lpar.rowwise() / _lpar.colwise().sum()).log();
}

Eigen::VectorXf
bagofwordsSampler::Item::operator()(int item) const
{
  Eigen::VectorXf out(s->nclus());
  for (int clus = 0; clus < s->nclus(); clus++) {
    out[clus] = (s->_lpar.col(clus).array() * s->_data.col(item).array().cast<float>()).sum();
  }
  return out;
}
