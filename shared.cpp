#include "shared.hpp"
#include "interdataset.hpp"

void
shared::sampleFromPrior()
{
  std::uniform_int_distribution<int> ainit(0,nclus()-1);

  for (auto i = 0; i < _alloc.cols(); i++)
    for (auto j = 0; j < _alloc.rows(); j++)
      _alloc(j, i) = ainit(generator);
}

void
shared::setAlloc(const Eigen::MatrixXi &x)
{
  assert(x.rows() == nitems() && x.cols() == nfiles());
  assert(x.minCoeff() >= 0 && x.maxCoeff() < nclus());
  _alloc = x;
}

void
shared::setAlloc(int k, Eigen::VectorXi file)
{
  assert(file.size() == nitems());
  assert(file.minCoeff() >= 0 && file.maxCoeff() < nclus());
  _alloc.col(k) = file;
}


float
shared::nuConditionalAlpha() const
{
  return nitems();
}

float
shared::phiConditionalAlpha(int k, int l) const
{
  return (_alloc.col(k).array() == _alloc.col(l).array()).count();
}

float
shared::gammaConditionalAlpha(int k, int j) const
{
  return 1 + (_alloc.col(k).array() == j).count();
}
