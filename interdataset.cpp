#include <unordered_map>

#include "shared.hpp"
#include "interdataset.hpp"

#include "cuda/cudasampler.hpp"

// calculate the index of item (i,j) in a (n x n) lower-triangular matrix
// calculate the product sums
interdataset::interdataset(const int K) : _nfiles(K)
{
  // ensure the artifact resulting from this implementation is
  // respected.  In practice, a K above about 8 will take a *long*
  // time to complete, so K=32 or 64 here doesn't matter.
  if ((size_t)K > sizeof(int)*8) {
    fprintf (stderr, "Error: too many data files (K=%i,max=%zu)", K, sizeof(int)*8);
    throw "fail!";
  }
  int
    m  = nphis(),
    m2 = 1<<m;

  std::vector<int> idx, set(K);

  if (true) {
    // this uses the sensible ordering of phis (i.e. the same as is used for MCMC output)
    for (int i = 0; i < K-1; i++) {
      for (int j = i+1; j < K; j++) {
        idx.push_back(i);
        idx.push_back(j);
      }
    }
  } else {
    // this is the order used in the Matlab MDI code—reversed for some reason!
    for (int i = K-2; i >= 0; --i) {
      for (int j = K-1; j > i; --j) {
        idx.push_back(i);
        idx.push_back(j);
      }
    }
  }

  std::unordered_map<std::vector<bool>,uint32_t> map;

  _weightord.push_back(0);
  _phiord.reserve(m2);

  std::vector<bool> key((1<<K)-1);

  for (int i = 0; i < m2; i++) {
    // zero out the key
    std::fill(key.begin(), key.end(), false);
    // assign every file to its own cluster
    for (int j = 0; j < K; j++) set[j] = j;
    int used = (1<<K)-1;
    // loop through phis we're interested in and merge clusters
    // according to which ones are together
    int x = i;
    // we want to loop through all set bits, so terminating condition
    // is when there aren't any left
    while (x) {
      // ffs() is "find first set" bit, so @j will iterate over set
      // bits only and @a and @b are the clusters we want to merge
      const int
	j = ffs(x)-1,
	a = set[idx[j * 2]],
	b = set[idx[j * 2 + 1]];
      // merge together
      for (int k = 0; k < K; k++) {
	if (set[k] == b)
	  set[k] = a;
      }
      // update our cluster used flags
      used = (used & ~(1<<b)) | (1<<a);
      // mask this entry off so we don't find it next time around
      x &= ~(1<<j);
    }
    // loop through all the used clusters
    while (used) {
      // get the first cluster
      int k = ffs(used)-1, l = 0;
      for (int j = 0; j < K; j++) {
	if (set[j] == k) {
	  l |= 1<<j;
	}
      }
      // add the sum in
      key[l-1] = true;
      // mask this cluster off as being used so we don't hit it next
      // time around
      used &= ~(1<<k);
    }

    int idx;
    auto got = map.find (key);
    if (got == map.end()) {
      idx = map.size();
      map.insert(std::make_pair(key,idx));

      // add this into our table so access can be FAST later
      int i = _weightord.size(), n = 0;
      _weightord.push_back(0);
      for (size_t j = 0; j < key.size(); j++) {
        if (key[j]) {
          _weightord.push_back(j);
          n++;
        }
      }
      _weightord[i] = n;
    } else {
      idx = got->second;
    }
    _phiord.push_back(idx);
  }

  _weightord[0] = map.size();

  _weightord.shrink_to_fit();
  _phiord.shrink_to_fit();
}

void
mdisampler::sampleFromPrior()
{
  /* TODO: This currently doesn't sample from the prior, just set of
   * independent Gammas which is not what the function name suggests,
   * but is fine for seeding parameters for a random walk */
  gamma_distribution<> ginit(2,1);

  for (auto i = 0; i < _dpmass.size(); i++)
    _dpmass(i) = ginit(generator);

  for (auto i = 0; i < _weight.cols(); i++)
    for (auto j = 0; j < _weight.rows(); j++)
      _weight(j, i) = ginit(generator);

  for (auto i = 0; i < _phi.size(); i++)
    _phi(i) = ginit(generator);

  _nu = ginit(generator);

  updateColumnProducts();
}

void
mdisampler::updateColumnProducts()
{
  for (int k = 0; k < nfiles(); k++) {
    const int k_ = (1<<k)-1;
    _prodsum(k_) = (_prod.col(k_) = _weight.col(k)).sum();
    for (int l = 0; l < k_; l++) {
      int l_ = k_+l+1;
      _prodsum(l_) = (_prod.col(l_) = _prod.col(k_).array() * _prod.col(l).array()).sum();
    }
  }
}

Eigen::VectorXd
mdisampler::calcProdCoef () const
{
  const auto& weightord = _inter.getWeightOrd();
  Eigen::VectorXd coef(weightord[0]);
  for (int i = 0, j = 1; i < coef.size(); i++) {
    double prod = 1;
    for (int n = weightord[j++]; n > 0; n--) {
      prod *= _prodsum(weightord[j++]);
    }
    assert(std::isfinite(prod));
    coef[i] = prod;
  }

  return coef;
}

double
mdisampler::calcProdPhi(const Eigen::VectorXd &coef) const
{
  const auto& phiord = _inter.getPhiOrd();
  int
    KK = nphis();
  double sum = 0;
  for (size_t i = 0; i < phiord.size(); i++) {
    double prod = coef[phiord[i]];
    for (int j = 0; j < KK; j++) {
      if(i & (1<<j))
        prod *= _phi(j);
    }
    assert(std::isfinite(prod));
    sum += prod;
  }
  return sum;
}

// calculate the normalisation constant from the given product vector
double
mdisampler::nuConditionalBeta() const
{
  if (nfiles() == 1)
    return _prodsum(0);

  return calcProdPhi(calcProdCoef());
}

// calculate the conditional distribution for phi_{k,l}
double
mdisampler::phiConditionalBeta (int k, int l) const
{
  const auto& phiord = _inter.getPhiOrd();

  int
    KK = nphis(),
    phiidx = phiBetween(k, l);

  Eigen::VectorXd coef = calcProdCoef();
  double sum = 0;
  for (size_t i = 0; i < phiord.size(); i++) {
    if (!(i & (1<<phiidx)))
      continue;
    double prod = coef[phiord[i]];
    for (int j = 0; j < KK; j++) {
      if(i & (1<<j))
        prod *= _phi(j);
    }
    sum += prod;
  }

  return _nu * sum / _phi(phiidx);
}

double
mdisampler::gammaConditionalBeta (int k, int clus) const
{
  if (nfiles() == 1)
    return _nu;

  const auto& weightord = _inter.getWeightOrd();
  Eigen::VectorXd coef(weightord[0]);
  for (int i = 0, j = 1; i < coef.size(); i++) {
    double prod = 1;
    for (int n = weightord[j++]; n > 0; n--) {
      int kk = weightord[j++];
      // if this product is affected by our dataset of interest
      if ((kk+1) & (1<<k)) {
	int kk1 = (kk+1) & ~(1<<k);
	for (int m = 0; m < nfiles(); m++) {
	  if(kk1 & (1<<m)) {
	    prod *= _weight(clus,m);
	  }
	}
      } else {
	// otherwise, just use the precalculated value
	prod *= _prodsum(kk);
      }
    }
    coef[i] = prod;
  }
  return _nu * calcProdPhi(coef);
}



// void
// mdisampler::setWeight(int j, int k, double weight)
// {
//   assert(weight >= 0);
//   _weight(j,k) = weight;

//   // should be able to optimise this a bit more!
//   updateColumnProducts();
// }

void
mdisampler::setDpMass(int k, double mass)
{
  assert(mass >= 0);
  _dpmass(k) = mass;
}

void
mdisampler::setPhi(int k, int l, double phi)
{
  assert(phi >= 0);

  _phi[phiBetween (k, l)] = phi;
}

void
mdisampler::setNu(double x)
{
  assert(x >= 0);
  _nu = x;
}

void
mdisampler::setWeight(const Eigen::MatrixXf &x)
{
  assert(x.rows() == nclus() && x.cols() == nfiles());
  assert(x.minCoeff() >= 0);
  _weight = x;
}

void
mdisampler::setWeight(int k, const Eigen::VectorXf &x)
{
  assert(x.minCoeff() >= 0);
  _weight.col(k) = x;
}

void
mdisampler::setDpMass(const Eigen::VectorXf &x)
{
  assert(x.size() == nfiles());
  _dpmass = x;
}

void
mdisampler::setPhis(const Eigen::VectorXf &x)
{
  assert(x.size() == nphis());
  _phi = x;
}
