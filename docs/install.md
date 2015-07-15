# Installing MDI++ #

This program has various build-time requirements, these are:

1. **C++11 compiler**, GCC or Clang have been tested under OSX and
   Ubuntu.

2. **GNU make**, limited GNU features are used, but I haven't tested
   with anything else.

3. **pkg-config**, used within `config.mk` to auto-detect the location
   of headers

4. **Boost**, only the `program_options` module is used to parse command
   line parameters at the moment.

5. **Eigen 3**, is used for all linear algebra within MDI++.

6. optional: **CUDA**, is used for enable GPU based computation.

What you have to do to get these various packages installed on your
computer will depend; I'll start by assuming that the reader is
running Ubuntu.

## Ubuntu ##

Installation under Ubuntu is easiest, after downloading the code,
running the following commands should suffice to build MDI++:

    $ sudo apt-get install build-essentials libboost-all-dev libeigen3-dev
	$ sudo apt-get install nvidia-cuda-toolkit  # only needed if GPU support wanted

and all should be OK.  If you want to use GPU acceleration, you will
need the CUDA toolkit to be installed (I'm currently using 6.5).
There are various ways of going about this, but for reference under
OSX I used `cuda_6.5.14_mac_64.pkg` from NVIDIA and under my recent
version of Ubuntu it is already included and can be installed with:

    $ sudo apt-get install nvidia-cuda-toolkit

Some modification of config.mk may be needed to get it building with
CUDA, there are some commented sections that work for my various
configurations.

## Building MDI++ ##

Should you not want to build with CUDA support, you can set the
preprocessor macro `ncuda`, for example:

    $ make ncuda=1

The makefiles also support `opt` for turning optimisations, due to
optimisations within Eigen it is strongly recommended to build with
this options when analysing larger datasets:

    $ make opt=1

the `ncuda` and `opt` flags can both be passed to turn off CUDA
support while turning on compiler optimisations.

## OS X ##

There are a variety of package managers available under OS X, common
alternatives are Fink, MacPorts and Homebrew, but I personally use
[Homebrew](http://brew.sh/) so the following instructions will target
this.  Note that you should expect to encounter incompatibilities if
you install more than one of the above systems.

Installing the required C++ libraries under Homebrew is similar to
Ubuntu:

    $ brew install --c++11 boost eigen

You will need a working compiler to install these packages, so it
doesn't need to be requested explicitly.  However, and unfortunately,
interactions with CUDA make the above simple recipe more complicated
due to it using a different `stdlib`.  The easiest way I have found of
getting CUDA support working under OSX is by using GCC.  You would
threfore run:

    $ brew install gcc-4.9
    $ brew install --c++11 --cc=gcc-4.9 --build-from-source boost eigen

which will install the compiler and then build and install the C++
libraries using the correct conventions.

## Redhat Linux ##

Redhat follows a similar naming convention to Ubuntu.  Boost and Eigen
can be installed with the following commands:

    # yum install boost boost-devel
    # yum install eigen3 eigen3-devel

then you probably need to get an update compiler installed as the
default included in RedHat is very out of date.  This can be
accomplished by running:

	# yum install devtoolset-2-toolchain

`devtoolset-3` is also available, and can be used if you are able to
use it.  Finally the build command is somewhat different, as you need
to run it "within" Redhat's Developer Toolset:

	$ scl enable devtoolset-2 'make ncuda=1'

We don't have an Nvidia card on this box so therefore can't comment on
installing CUDA inside Redhat.  If you have success please let me know
and I'll update this document and other people can benefit.