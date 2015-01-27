//
//  dataset.hpp
//  gencoefficents
//
//  Created by Sam Mason on 04/02/2014.
//  Copyright (c) 2014 Sam Mason. All rights reserved.
//

#ifndef mdi_dataset_hpp
#define mdi_dataset_hpp

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include "csv.hpp"
#include "interdataset.hpp"
#include "cuda/cudasampler.hpp"

class shared {
  int _nclus;
  Eigen::MatrixXi _alloc;

public:
  shared(int K, int N, int n) : _nclus(N), _alloc(Eigen::MatrixXi::Zero(n, K)) { }

  int nitems() const { return _alloc.rows();  }
  int nfiles() const { return _alloc.cols();  }
  int nclus()  const { return _nclus; }

  // sample all state from prior --- good for initialisation!
  void sampleFromPrior();

  const Eigen::MatrixXi &allocs() const { return _alloc;  }
  typename Eigen::MatrixXi::ConstColXpr alloc(int k)  const {
    return _alloc.col(k);  }

  void setAlloc(const Eigen::MatrixXi &x);
  void setAlloc(int k, Eigen::VectorXi file);

  int alloc(int k, int i) const { return _alloc(i, k); }

  float	nuConditionalAlpha() const;
  float	phiConditionalAlpha(int k, int l) const;
  float	gammaConditionalAlpha(int k, int j) const;
};

#endif
