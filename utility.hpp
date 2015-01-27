template<class InputIt> void
printList (InputIt first, InputIt last)
{
  while(first != last)
    std::cerr << *first++ << ' ';
  std::cerr << '\n';
}

template<int Flags=Eigen::ColMajor, typename Derived>
std::vector<typename Derived::Scalar>
eigenMatrixToStdVector(const Eigen::DenseBase<Derived> &m) {
  typedef typename Derived::Scalar Scalar;
  std::vector<Scalar> out(m.size());
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Flags> >
    (out.data(), m.rows(), m.cols()) = m;
  return out;
}

template <int Flags=Eigen::ColMajor, typename Scalar>
Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Flags> >
stdVectorToEigenMatrix(const std::vector<Scalar> &v, int rows, int cols)
{
  assert((size_t)(rows*cols) == v.size());
  return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Flags> >
    (v.data(), rows, cols);
}

template <int Flags=0, typename Scalar>
Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Flags> >
stdVectorToEigenVector(const std::vector<Scalar> &v, int length)
{
  assert((size_t)length == v.size());
  return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Flags> >
    (v.data(), length);
}
