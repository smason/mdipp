include config.mk

SUBDIRS := cuda datatypes

.SUFFIXES: .c .cpp .cu .o .cu.o
.PHONY: all clean $(SUBDIRS)

OBJS := interdataset.o csv.o csvpp.o shared.o datatypes/datatypes.a

ifndef ncuda
OBJS := $(OBJS) cuda/cudasampler.a
endif

# Phony targets
all: mdi++

test: runtests
	./runtests

clean:
	rm -f multiphi multiphi.o
	rm -f runtests runtests.o
	rm -f mdi++ mdi++.o csvpp.o shared.o interdataset.o csv.o gencoefficents.o
	make -C datatypes clean
	make -C cuda      clean

# Subdirectories
cuda:
	make -C cuda opt=$(opt) ndebug=$(ndebug) ncuda=$(ncuda) cudasampler.a

datatypes:
	make -C datatypes opt=$(opt) ndebug=$(ndebug) ncuda=$(ncuda) datatypes.a

# Executables
multiphi: multiphi.o

gencoefficents: gencoefficents.o csv.o

mdi++: mdi++.o $(OBJS)
	$(LINKER) $(LDFLAGS) -o $@ $^ $(LIBS)

runtests: runtests.o $(OBJS)
	$(LINKER) $(LDFLAGS) -o $@ $^ $(LIBS)

# Libraries
cuda/cudasampler.a: cuda

datatypes/datatypes.a: datatypes

# awkward one! I have a csv.c and a csv.cpp
csvpp.o: csv.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^
