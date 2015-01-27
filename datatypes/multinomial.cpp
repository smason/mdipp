#include <map>
#include <set>

#include "csv.hpp"
#include "datatypes.hpp"
#include "utility.hpp"

multinomSampler::multinomSampler(class cuda::sampler *gpu,
				 const Eigen::MatrixXi &data,
				 const std::vector<int> &levels, int nclus)
  : _nclus(nclus), _data(data), _levels(levels)
{
  _accum.reserve(levels.size());
  _lpar.reserve(levels.size());

  for (auto n : levels) {
    _accum.emplace_back(Eigen::MatrixXi::Zero(n, nclus));
    _lpar.emplace_back(Eigen::MatrixXf::Zero(n, nclus));
  }

  if(gpu) {
#ifndef NOCUDA
    int nlevels = 0;
    for (auto n : levels)
      nlevels = std::max(nlevels, n);

    _gpu.gpu = gpu->nextFileSampler();
    _gpu.nfeatures = nfeatures();
    _gpu.nlevels   = nlevels;

    _gpu.d_data.resize(nitems() * nfeatures());
    _gpu.d_lpar.resize(nclus * nfeatures() * nlevels);

    _gpu.d_data = eigenMatrixToStdVector(data.transpose());
#endif
  } else {
    _gpu.gpu = 0;
    _gpu.nfeatures = 0;
    _gpu.nlevels   = 0;
  }
}

void
multinomSampler::sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc)
{
  for (auto &x : _accum) x.setZero();
  for (int i = 0; i < nitems(); i++) {
    int j = alloc(i);
    for (int f = 0; f < nfeatures(); f++) {
      _accum[f](_data(f, i), j) += 1;
    }
  }
  for (int f = 0; f < nfeatures(); f++) {
    _lpar[f] = _accum[f].array().cast<float>().unaryExpr([] (float n) {
        return gamma_distribution<>(0.5 + n, 1)(generator);
      });
    for (int i = 0; i < _lpar[f].cols(); i++) {
      _lpar[f].col(i) = (_lpar[f].col(i).array() / _lpar[f].col(i).sum()).log();
    }
  }
}

Eigen::VectorXf
multinomSampler::Item::operator()(int item) const
{
  Eigen::VectorXf out = Eigen::VectorXf::Zero(s->nclus());
  for (int f = 0; f < s->nfeatures(); f++) {
    out += s->_lpar[f].row(s->_data(f,item));
  }
  return out;
}

multinomDatatype::multinomDatatype(const char *path)
{
  _data = loadNamedRectCsv(path, _colnames, _rownames, stoi_strict).transpose();
  const int
    nfeatures = _data.rows(),
    nitems    = _data.cols();
  
  _levels.reserve(nfeatures);

  for (int f = 0; f < nfeatures; f++) {
    std::set<int> s;
    for (int j = 0; j < nitems; j++) {
      s.insert(_data(f,j));
    }
    std::map<int,int> m;
    for (auto n : s)
      m.emplace(n, m.size());
    _levels.push_back(s.size());
    for (int i = 0; i < nitems; i++) {
      _data(f,i) = m[_data(f,i)];
    }
  }
}
