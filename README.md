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

    $ mdi++ N input.csv > demo/mcmc_01.csv

As can hopefully be seen from this command line, the list of input
files is specified and prefixed by their "data type", with standard
output being piped to a file where it can be used for plotting or
subsequent analysis.

The majority of the implementation of MDI++ is in C++ with compute
intensive portions employing CUDA.  A number of R scripts are provided
for the purposes of plotting and extracting a canonical clustering.
As with most Unix programs, a synopsis of command line options can be
displayed by running either:

    $ mdi++ --help

There are a number of R scripts in the "scripts" subdirectory that can
be used to load the CSV files MDI++ generates.  Once the CSV files
have been loaded various analyses and plots can be generated to help
you understand the clustering.  Inside R or RStudio, plots can be
generated

    > source('scripts/analysis.R')
    > dta <- loadDataFilesFromPath('demo/input.csv')
    > out <- readMdiMcmcOutput('demo/mcmc_01.csv')
    > cpsm <- generateConsensusPSM(out)
    > plotConsensusPSM(cpsm, dta, )

This program has various build-time requirements, namely a recent
build of the C++ libraries Boost and Eigen, and a recent C++11
compiler such as GCC 4.9 or clang.  Note, that in order to run with
CUDA you'll probably need to use gcc as that was the only way I could
get libstdc++ bindings correct---patches welcome to fix this!  The
code has been developed under both Ubuntu and OS X and should just
work with either, I don't use Windows at the moment so it may require
significant work to build/use it there.

# Building #

I develop under Ubuntu and OSX, so these should be well supported.
The code is written in C++ and uses some C++11 extensions, so a
somewhat recent compiler is needed, I use GCC-4.9.  A couple of
external libraries are used: Boost and Eigen.  On OSX, these are
somewhat complicated by the choice of standard libraries (or were when
I started work and haven't checked, simplifications welcome!) and
hence I use homebrew to install these as:

    $ brew install --c++11 --cc=gcc-4.9 --build-from-source boost eigen

where gcc-4.9 is the compiler you're using--on Linux everything just
works, so no need to worry there, just do the normal:

    $ sudo apt-get install libboost-all-dev libeigen3-dev

and all should be OK.  If you want to use GPU acceleration, you will
need the CUDA toolkit to be installed (I'm currently using 6.5).
There are various ways of going about this, but for reference under
OSX I used cuda_6.5.14_mac_64.pkg from NVIDIA and under my recent
version of Ubuntu it is already included and can be installed with:

    $ sudo apt-get install nvidia-cuda-toolkit

Some modification of config.mk may be needed to get it building with
CUDA, there are some commented sections that work for my various
configurations.

Should you not want to build with CUDA support, you can set the
preprocessor macro "ncuda", for example:

    $ make ncuda=1

The makefiles also support "opt" for turning optimisations on, and
"ndebug" for turning off debug support.  To build for highest
performance, you can therefore build with:

    $ make ndebug=1 opt=1

although I'd recommend not turning off debug information.