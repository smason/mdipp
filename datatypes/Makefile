include ../config.mk

CXXFLAGS := $(CXXFLAGS) -I..

clean:
	rm -f datatypes.a bagofwords.o gaussian.o gaussianprocess.o multinomial.o

datatypes.a: bagofwords.o gaussian.o gaussianprocess.o multinomial.o
	$(ARCHIVER) $@ $^
