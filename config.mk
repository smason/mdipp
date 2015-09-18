# OSX install:
#   brew install --c++11 --cc=gcc-4.9 --build-from-source boost eigen
# Ubuntu install:
#   sudo apt-get install libboost-all-dev libeigen3-dev nvidia-cuda-toolkit

EIGEN := $(shell pkg-config eigen3 --cflags)
BOOST :=

LINKER   ?= $(CXX)
ARCHIVER ?= ar -rus
NVCC     ?= nvcc -arch=sm_21 # -Xptxas="-v"

CFLAGS   ?= -std=c99 -Wall
CXXFLAGS ?= -std=c++11 -Wall $(EIGEN)

LIBS := -lboost_program_options # -static-libgcc -static-libstdc++

ifdef ncuda
  CFLAGS   := $(CFLAGS)   -DNOCUDA
  CXXFLAGS := $(CXXFLAGS) -DNOCUDA
  NVCC     := true
else
# for Ubuntu
  # CFLAGS  := $(CFLAGS)  $(shell pkg-config cudart-6.5 --cflags)
  # LDFLAGS := $(LDFLAGS) $(shell pkg-config cudart-6.5 --libs)
  # LINKER  := nvcc

# for OSX
  LDFLAGS := $(LDFLAGS) -L/Developer/NVIDIA/CUDA-6.0/lib -lcudart
endif

ifndef ndebug
  CFLAGS    := $(CFLAGS)    -g -gdwarf-2
  CXXFLAGS  := $(CXXFLAGS)  -g -gdwarf-2
  LDFLAGS   := $(LDFLAGS)   -g -gdwarf-2
  NVFLAGS   := $(NVFLAGS)   -g -G
  NVLDFLAGS := $(NVLDFLAGS) -g -G
endif

ifdef opt
  CFLAGS    := $(CFLAGS)    -ffast-math -O3 -mtune=native -DNDEBUG
  CXXFLAGS  := $(CXXFLAGS)  -ffast-math -O3 -mtune=native -DNDEBUG
  LDFLAGS   := $(LDFLAGS)   -ffast-math -O3 -mtune=native
  NVFLAGS   := $(NVFLAGS)   -O3 -DNDEBUG
  NVLDFLAGS := $(NVLDFLAGS) -O3
endif
