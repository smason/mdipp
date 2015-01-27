#ifndef mdi_interdataset_hpp
#define mdi_interdataset_hpp

#include <vector>
#include "stats.hpp"

#include <Eigen/Dense>

class interdataset {
  int _nfiles;
  std::vector<int> _phiord;    // was first
  std::vector<int> _weightord; // was second

public:
  interdataset(int nfiles);

  int nfiles() const { return _nfiles; }
  int nphis()  const { return _nfiles*(_nfiles-1)/2; }

  const std::vector<int> &getPhiOrd()    const { return _phiord; }
  const std::vector<int> &getWeightOrd() const { return _weightord; }
};

class mdisampler {
  const interdataset &_inter;

  Eigen::MatrixXf _weight;
  Eigen::VectorXf _dpmass, _phi;
  double _nu;

  Eigen::MatrixXf _prod;
  Eigen::VectorXf _prodsum;

  void updateAllocHistogram();
  void updateColumnProducts();

  Eigen::VectorXd calcProdCoef() const;
  double calcProdPhi(const Eigen::VectorXd &coef) const;

public:
  mdisampler(const interdataset &inter, int nclus)
    : _inter(inter),
      _weight(nclus, inter.nfiles()),
      _dpmass(Eigen::VectorXf::Zero(inter.nfiles())),
      _phi(Eigen::VectorXf::Zero(inter.nphis())),
      _nu(0.0),
      _prod(nclus, (1<<inter.nfiles())-1), _prodsum((1<<inter.nfiles())-1)
  { }

  int nclus()  const { return _weight.rows(); }
  int nfiles() const { return _weight.cols(); }
  int nphis()  const { return _inter.nphis(); }

  void setDpMass(int k, double mass);
  void setWeight(int k, const Eigen::VectorXf &weight);
  void setWeight(const Eigen::MatrixXf &x);
  void setPhi(int k, int l, double phi);
  void setNu(double x);

  // void setWeight(int j, int k, double weight);
  void setDpMass(const Eigen::VectorXf &x);
  void setPhis(const Eigen::VectorXf &x);

  double dpmass(int k)        const { return _dpmass(k); }
  double phi(int phi_kl)      const { return _phi(phi_kl); }
  double phi(int k, int l)    const { return _phi(phiBetween(k, l)); }
  double nu()                 const { return _nu; }
  double weight(int j, int k) const { return _weight(j, k); }

  const Eigen::VectorXf &dpmass()  const { return _dpmass; }
  const Eigen::MatrixXf &weights() const { return _weight; }
  const Eigen::VectorXf &phis()    const { return _phi;    }
  typename Eigen::MatrixXf::ConstColXpr weight(int k) const {
    return _weight.col(k); }

  Eigen::VectorXf phis(int k) const {
    Eigen::VectorXf out(nfiles());

    for (int j = 0; j < nfiles(); j++) {
      if (j < k)
        out[j] = phi(j, k);
      else if (j > k)
        out[j] = phi(k, j);
      else
        out[j] = NAN;
    }

    return out;
  }

  int phiBetween (int k, int l) const {
    assert(k < nfiles() && l < nfiles() && k < l);
    return nfiles()*k - (k+1)*k/2 + l-k - 1;
  }

  void sampleFromPrior();

  double nuConditionalBeta() const;
  double phiConditionalBeta(int k, int l) const;
  double gammaConditionalBeta(int k, int j) const;

  #ifndef NOCUDA

  #endif
};

#endif
