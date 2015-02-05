#ifndef mdi_datatypes_hpp
#define mdi_datatypes_hpp

#include <memory>

#include "cuda/cudasampler.hpp"

#include "shared.hpp"
#include "stats.hpp"

class sampler {
public:
  virtual ~sampler() {};

  // cluster allocations have been altered from i to changed[i], by
  // default we assume that all parameters are Gibbs sampled during
  // the next call to @sampleParams
  virtual void swap(const std::vector<int> &changed) { };

  virtual void sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc) = 0;

  // code for calculating the log-likelihoods of any given item being
  // assigned to each cluster, given current cluster parameters.  it's
  // in a functor so that calculations can be reused appropriately
  struct item {
    virtual ~item() {}
    virtual Eigen::VectorXf operator()(int i) const = 0;
  };

  virtual item* newItemSampler() const = 0;

  virtual void sampleFeatureSelection(const Eigen::Ref<const Eigen::VectorXi> &alloc) {  }

  virtual Eigen::VectorXi featureState() const { return Eigen::VectorXi::Zero(0); }

#ifndef NOCUDA
  /** sample the cluster parameters given the specified cluster
   * allocations. note that GPU version of @alloc will be maintained
   * outside this API
   */
  virtual void cudaSampleParameters(const Eigen::Ref<const Eigen::VectorXi> &alloc) = 0;
  /** accumulate the log-probability of cluster associations into
   * cuda::datatype::myprobs(), @cudaSampleParameters will have been
   * called prior to this
   */
  virtual void cudaAccumAllocProbs() = 0;
#endif
};

class datatype {
public:
  virtual ~datatype() {};

  virtual std::vector<std::string> items()    const = 0;
  virtual std::vector<std::string> features() const = 0;

  virtual sampler* newSampler(int nclus, class cuda::sampler *gpu) const = 0;
};

template<class T, T fn(const std::string&)>
class csvdatastore : public datatype {
protected:
  // _rownames and _colnames must be before _data as they're referred to in its construction
  std::vector<std::string> _rownames, _colnames;
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _data;

public:
  csvdatastore(const char *path)
    : _data(loadNamedRectCsv(path, _colnames, _rownames, fn).transpose()) {}

  const Eigen::MatrixXf & rawdata() const { return _data; }

  std::vector<std::string> items()    const { return _rownames; }
  std::vector<std::string> features() const { return std::vector<std::string>(); }
};

class gaussianSampler : public sampler {
  int _nclus;
  const Eigen::MatrixXf _data;

  // @_mu are the means, @_tau are the precisions (i.e. 1/sqrt(_tau) = sd)
  Eigen::MatrixXf _mu, _tau;
  Eigen::VectorXi _featurestate;

  class Item : public item {
    const gaussianSampler *s;
  public:
    Item(const gaussianSampler *s) : s(s) {}
    Eigen::VectorXf operator()(int item) const;
  };

  Eigen::ArrayXf logprobItemClus(int item, int clus) const;

public:
  gaussianSampler(class cuda::sampler *gpu, const Eigen::MatrixXf &data, int nclus);

  int nitems()    const { return _data.cols(); }
  int nfeatures() const { return _data.rows(); }
  int nclus()     const { return _nclus;       }

  // not private for testing purposes
  std::vector<runningstats<> > accumState(const Eigen::Ref<const Eigen::VectorXi> &alloc);

  void debug_setMuTau(Eigen::MatrixXf mu, Eigen::MatrixXf tau) {
    _mu  = mu;
    _tau = tau;
  }

  void sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc);

  Item* newItemSampler() const {
    return new Item(this);
  }

  void sampleFeatureSelection(const Eigen::Ref<const Eigen::VectorXi> &alloc);

  Eigen::VectorXi featureState() const { return _featurestate; }

#ifndef NOCUDA
private:
  cuda::gaussian _gpu;

public:
  void cudaSampleParameters(const Eigen::Ref<const Eigen::VectorXi> &alloc) { _gpu.sampleParameters(); }
  void cudaAccumAllocProbs() { _gpu.accumAllocProbs(); }
#endif
};

class gaussianDatatype : public csvdatastore<float, stof_strict> {
public:
  gaussianDatatype(const char *path) : csvdatastore(path) {}

  std::vector<std::string> features() const { return _colnames; }

  gaussianSampler* newSampler(int nclus, class cuda::sampler *gpu) const {
    return new gaussianSampler(gpu, _data, nclus);
  }
};


class bagofwordsSampler : public sampler {
  int _nclus;
  const Eigen::MatrixXi &_data;

  Eigen::ArrayXXf _lpar;

  class Item : public item {
    const bagofwordsSampler *s;
  public:
    Item(const bagofwordsSampler *s) : s(s) {}
    Eigen::VectorXf operator()(int item) const;
  };

public:
  bagofwordsSampler(class cuda::sampler *gpu, const Eigen::MatrixXi &data, int nclus);

  int nitems()    const { return _data.cols(); }
  int nfeatures() const { return _data.rows(); }
  int nclus()     const { return _nclus; }

  void sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc);

  Item* newItemSampler() const {
    return new Item(this);
  }

#ifndef NOCUDA
private:
  cuda::bagofwords _gpu;

public:
  void cudaSampleParameters(const Eigen::Ref<const Eigen::VectorXi> &alloc) { _gpu.sampleParameters(); }
  void cudaAccumAllocProbs() { _gpu.accumAllocProbs(); }
#endif
};

class bagofwordsDatatype : public csvdatastore<int, stoi_strict> {
public:
  bagofwordsDatatype(const char *path) : csvdatastore(path) {}

  bagofwordsSampler* newSampler(int nclus, class cuda::sampler *gpu) const {
    return new bagofwordsSampler(gpu, _data, nclus);
  }
};


class multinomSampler : public sampler {
  int _nclus;
  const Eigen::MatrixXi &_data;
  const std::vector<int> &_levels;

  std::vector<Eigen::ArrayXXf> _lpar;

  class Item : public item {
    const multinomSampler *s;
  public:
    Item(const multinomSampler *s) : s(s) {}
    Eigen::VectorXf operator()(int item) const;
  };

public:
  multinomSampler(class cuda::sampler *gpu, const Eigen::MatrixXi &data,
		  const std::vector<int> &levels, int nclus);

  int nitems()    const { return _data.cols(); }
  int nfeatures() const { return _data.rows(); }
  int nclus()     const { return _nclus; }

  void sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc);

  Item* newItemSampler() const {
    return new Item(this);
  }

#ifndef NOCUDA
private:
  cuda::multinomial _gpu;

public:
  void cudaSampleParameters(const Eigen::Ref<const Eigen::VectorXi> &alloc) { _gpu.sampleParameters(); }
  void cudaAccumAllocProbs() { _gpu.accumAllocProbs(); }
#endif
};

class multinomDatatype : public datatype {
  // _rownames and _colnames must be before _data as they're referred to in its construction
  std::vector<std::string> _rownames, _colnames;
  std::vector<int> _levels;
  Eigen::MatrixXi _data;

public:
  multinomDatatype(const char *path);

  std::vector<std::string> items()    const { return _rownames; }
  std::vector<std::string> features() const { return _colnames; }

  multinomSampler* newSampler(int nclus, class cuda::sampler *gpu) const {
    return new multinomSampler(gpu, _data, _levels, nclus);
  }
};


/* We refer to @sf, @sn and @l throughout as the hyperparameters of a GP.
 * These are the standard deviation of the function, the additive noise and
 * characteristic length scale of a squared exponential covariance function.
 *
 * Often we have @pf, @pn, @pl, which refers to the "precision" of the
 * function, noise and length scales (i.e. @pf = 1/@sf^2)
 */

class gpHypers {
  gamma_distribution<double> _pf, _pn, _pl;

public:
  gpHypers () : _pf(2.0,  0.5), _pn(2.0, 0.1), _pl(2.0, 0.5) { }

  gamma_distribution<double> pf() const { return _pf; }
  gamma_distribution<double> pn() const { return _pn; }
  gamma_distribution<double> pl() const { return _pl; }
};

class gp {
  double _pf, _pn, _pl;

public:
  gp(double pf, double pn, double pl) : _pf(pf), _pn(pn), _pl(pl) {}

  gp(const gpHypers &prior) { sampleFromPrior(prior); }

  double pf() const { return _pf; }
  double pn() const { return _pn; }
  double pl() const { return _pl; }

  void pf(double x) { _pf = x; }
  void pn(double x) { _pn = x; }
  void pl(double x) { _pl = x; }

  Eigen::MatrixXd covarianceMatrix (const Eigen::VectorXd &time) const;
  void sampleFromPrior (const gpHypers &prior);
  Eigen::VectorXd sampleFunction (const Eigen::VectorXd &time);
};

class gpSampler : public sampler {
  int _nclus;
  const Eigen::MatrixXf &_data;
  const Eigen::VectorXd &_time;

  std::vector<gp> _gp;

  gpHypers _prior;

  // our sampled functions
  Eigen::MatrixXf _sampfn;

  class Item : public item {
    const gpSampler *s;
    Eigen::ArrayXf prob0, prob1;
  public:
    Item(const gpSampler *s);
    Eigen::VectorXf operator()(int item) const;
  };

public:
  gpSampler(class cuda::sampler *gpu, const Eigen::MatrixXf &data,
	    const Eigen::VectorXd &time, int nclus);

  int nitems()    const { return _data.cols(); }
  int nfeatures() const { return _data.rows(); }
  int nclus()     const { return _nclus; }

  void swap(const std::vector<int> &changed);

  void sampleParams (const Eigen::Ref<const Eigen::VectorXi> &alloc);

  Item* newItemSampler() const {
    return new Item(this);
  }

#ifndef NOCUDA
private:
  cuda::gaussianprocess _gpu;

public:
  void cudaSampleParameters(const Eigen::Ref<const Eigen::VectorXi> &alloc);
  void cudaAccumAllocProbs() { _gpu.accumAllocProbs(); }
#endif
};

class gpDatatype : public csvdatastore<float, stof_strict> {
  Eigen::VectorXd _time;

public:
  gpDatatype(const char *path);

  gpSampler* newSampler(int nclus, class cuda::sampler *gpu) const {
    return new gpSampler(gpu, _data, _time, nclus);
  }
};

#endif // #ifndef mdi_datatypes_hpp
