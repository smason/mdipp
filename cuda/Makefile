include ../config.mk

NVFLAGS := $(NVFLAGS) -I..

.SUFFIXES: .c .cpp .cu .o .cu.o

all: cudasampler.a

clean:
	rm -f cudasampler.cu.o datatypes.cu.o cudasampler.a

%.o : %.cpp
	$(NVCC) $(NVFLAGS) -c -o $@ $<

%.cu.o : %.cu
	$(NVCC) $(NVFLAGS) -c -o $@ -Iinclude $<

cudasampler.a: cudasampler.cu.o datatypes.cu.o
	$(ARCHIVER) $@ $^

benchmark: benchmark.cu
	$(NVCC) $(NVFLAGS) -O -o $@ -DRNG_MODE=$(RNG_MODE) -Iinclude $^

cudasampler.cu.o: cudasampler.hpp

datatypes.cu.o: cudasampler.hpp samsrng.hpp
