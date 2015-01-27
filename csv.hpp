#ifndef gencoefficents_shared_hpp
#define gencoefficents_shared_hpp

#include <vector>

#include <Eigen/Dense>

extern "C" {
#include <stdio.h>
#include "csv.h"
}

double stod_strict(const std::string& str);
float stof_strict(const std::string& str);
int stoi_strict(const std::string& str);

template<typename T, int _Rows=Eigen::Dynamic, int _Cols=Eigen::Dynamic>
Eigen::Matrix<T, _Rows, _Cols>
loadRectCsv(const char *path, T (*conv)(const std::string&)) {
  FILE *fd = fopen (path, "r");

  if (!fd)
    throw "unable to open file";

  csvfile csv;
  csv_parse(&csv, fd);

  size_t nfields = 0;
  const char **row = csv_readrow (&csv, &nfields);
  if (!row || nfields < 1) {
    abort();
  }

  std::vector<T> out;
  out.reserve(nfields * 10);
  for (size_t i = 0; i < nfields; i++)
    out.push_back(conv(row[i]));

  size_t k;
  while ((row = csv_readrow (&csv, &k))) {
    if (k != nfields)
      abort();
    for (size_t i = 0; i < nfields; i++) {
      out.push_back(conv(row[i]));
    }
  }

  csv_close (&csv);

  typedef Eigen::Matrix<T, _Rows, _Cols, Eigen::RowMajor> MatrixXR;
  return Eigen::Map<MatrixXR>(out.data(), out.size() / nfields, nfields);
}

template<typename T, int _Rows=Eigen::Dynamic, int _Cols=Eigen::Dynamic>
Eigen::Matrix<T, _Rows, _Cols>
loadNamedRectCsv(const char *path,
                 std::vector<std::string> &colnames,
                 std::vector<std::string> &rownames,
                 T (*conv)(const std::string&)) {
  FILE *fd = fopen (path, "r");

  if (!fd)
    throw "unable to open file";

  csvfile csv;
  csv_parse(&csv, fd);

  colnames.clear();
  rownames.clear();

  size_t nfields = 0;
  const char **row = csv_readrow (&csv, &nfields);
  if (!row || nfields < 1) {
    abort();
  }

  colnames.reserve(nfields-1);
  colnames.reserve(16);
  std::vector<T> out;
  out.reserve(nfields * rownames.capacity());

  for (size_t i = 1; i < nfields; i++)
    colnames.push_back(row[i]);

  while ((row = csv_readrow (&csv, &nfields))) {
    if (nfields != colnames.size()+1)
      abort();
    rownames.push_back(row[0]);
    for (size_t i = 1; i < nfields; i++) {
      out.push_back(conv(row[i]));
    }
  }

  csv_close (&csv);

  typedef Eigen::Matrix<T, _Rows, _Cols, Eigen::RowMajor> MatrixXR;
  return Eigen::Map<MatrixXR>(out.data(), rownames.size(), colnames.size());
}

#endif
