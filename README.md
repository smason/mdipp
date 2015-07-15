# MDI++ — Multiple Dataset Integration, reimplemented #

MDI is a method for performing integrative clustering of genomic
datasets.  By "integrative" we mean that it is able to share
clustering correlations across multiple related datasets.  The
clustering model is probabilistic, and it is therefore able to find a
"natural" number of clusters to represent your data.  Furthermore a
Bayesian framework is used, so the output of analysis is a Monte Carlo
chain targeting the posterior distribution of the model parameters
given the data.

To see the simple case of clustering a single normally distributed
dataset, run:

    $ mdi++ N input.csv > mcmc_01.csv

As can hopefully be seen from this command line, the list of input
files is specified and prefixed by their "data type", with standard
output being piped to a file where it can be used for plotting or
subsequent analysis.

The majority of the implementation of MDI++ is in C++ with compute
intensive portions employing CUDA.  A number of R scripts are provided
for the purposes of plotting and extracting a canonical clustering.
As with most Unix programs, a synopsis of command line options can be
displayed by running:

    $ mdi++ --help

There are a number of R scripts in the `scripts` subdirectory that can
be used to load the CSV files MDI++ generates, the `demo` subdirectory
contains example data and analysis scripts, all further documentation
is included in the `docs` subdirectory.

# Building #

This software has been primarily developed under Mac OSX and Ubuntu,
so these are currently best supported.  The software dependencies are
a C++11 compiler (such as GCC or Clang) and the [Boost] and [Eigen]
libraries.  These can be installed under OSX, assuming you are not
using CUDA features, by running:

    $ brew install --c++11 boost eigen

or under Ubuntu by running:

    $ sudo apt-get install libboost-all-dev libeigen3-dev

Once these dependencies have been installed, one should be able to
type `make` and the software will build. There are some system
specific dependencies that are defined in `config.mk` that may need to
be tweaked depending on your system, and for more details see
[`docs/install.md`][install].

[Boost]: http://www.boost.org/
[Eigen]: http://eigen.tuxfamily.org/
[install]: docs/install.md
