## Makefile for building CUDA/CPP programs
## make nocuda for platforms without CUDA
## make perf for just the optimised (performance) executables
## make debug for debuggable executables
## make clean to clean up executables, object files, backup files and profile outputs

## written by Julian Waton

## to do...add Windows compatibility

SHELL = /bin/sh

## CPP variables
CC = gcc
CFLAGS = -Wall -Wextra -fopenmp
G = -g3 -pg
OPT = -O3 -ffast-math

SRC = utils.cpp wavutils.cpp
WSRC = haar.cpp daub4.cpp la8.cpp c6.cpp transform.cpp
HEADS = utils.h wvtheads.h wavutils.h transform.h
OS = utils.o wavutils.o
WOS = daub4.o haar.o la8.o c6.o thresh.o
TOS = transform.o
LIBS = -lm -lstdc++
EXES = cptesthaar cptestdaub4 cptestcvt cptestla8 cptestc6 cptesttransform
PROF = gmon.out


## CUDA variables
CUC = nvcc
CUFLAGS = -arch=sm_30 -m 64 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fopenmp
CUG = -g -G -Xcompiler -g -pg
CUOPT = -use_fast_math
CULINK = -dc

CUMAINS = testdaub4cuda.cu testhaarcuda.cu timehaarcuda.cu tabletimehaar.cu testla8cuda.cu testc6cuda.cu
CUSRC = utilscuda.cu
CUWSRC = haarcuda.cu daubcuda4.cu la8cuda.cu c6cuda.cu transformcuda.cu
CUHEADS = haarcuda.cuh utilscuda.cuh cudaheads.h daub4cuda.cuh threshcuda.cuh la8cuda.cuh wavutilscuda.cuh c6cuda.cuh transformcuda.cuh
CUOS = utilscuda.o wavutilscuda.o
CUWOS = haarcuda.o daub4cuda.o la8cuda.o c6cuda.o threshcuda.o
CUTOS = transformcuda.o
CULIBS = -l curand
CUEXES = cutesthaar cutestdaub4 cutimehaar cutimedaub4 cutabhaar cutabdaub4 cutestla8 cutestc6 cutestcvt


## Debug-mode variable.

OSDEB = utilsdebug.o wavutilsdebug.o
WOSDEB = daub4debug.o haardebug.o la8debug.o c6debug.o threshdebug.o
TOSDEB = transformdebug.o
EXESDEB = cptesthaard cptestdaub4d cptestcvtd cptestla8d cptestc6d cptestcvtd cptesttransformd

CUOSDEB = utilscudadebug.o wavutilscudadebug.o
CUWOSDEB = haarcudadebug.o daub4cudadebug.o la8cudadebug.o c6cudadebug.o threshdebug.o
CUTOSDEB = transformcudadebug.o
CUEXESDEB = cutesthaard cutestdaub4d cutimedaub4d cutabhaard cutabdaub4d cutestla8d cutestc6d cutimehaard cutestcvtd


.PHONY: all
all: $(EXES) $(CUEXES) $(CUEXESDEB) $(EXESDEB)

perf: $(EXES) $(CUEXES)

debug: $(CUEXESDEB) $(EXESDEB)

cuda: $(CUEXES) $(CUEXESDEB)

nocuda: $(EXES) $(EXESDEB)

cutab%: $(CUOS) $(OS) tabletime%.cu %cuda.o %.o
	$(CUC) $(CUFLAGS) $(CUOPT) -o $@ $^ $(CULIBS)

cutab%d: $(CUOSDEB) $(OSDEB) tabletime%.cu %cudadebug.o %debug.o
	$(CUC) $(CUFLAGS) $(CUG) -o $@ $^ $(CULIBS)

cutime%: $(CUOS) $(OS) time%cuda.cu %cuda.o %.o
	$(CUC) $(CUFLAGS) $(CUOPT) -o $@ $^ $(CULIBS)

cutime%d: $(CUOSDEB) $(OSDEB) time%cuda.cu %cudadebug.o %debug.o
	$(CUC) $(CUFLAGS) $(CUG) -o $@ $^ $(CULIBS)

cutest%: $(CUOS) $(OS) %cuda.o %.o test%cuda.cu 
	$(CUC) $(CUFLAGS) $(CUOPT) -o $@ $^ $(CULIBS)

cutest%d: $(CUOSDEB) $(OSDEB) %cudadebug.o %debug.o test%cuda.cu
	$(CUC) $(CUFLAGS) $(CUG) -o $@ $^ $(CULIBS)

cptest%: $(OS) %.o test%.cpp
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LIBS)

cptest%d: $(OSDEB) %debug.o test%.cpp
	$(CC) $(CFLAGS) $(G) -o $@ $^ $(LIBS)

cutestcvt: testcvtcuda.cu thresh.o threshcuda.o $(CUOS) $(CUWOS) $(OS) $(WOS) $(TOS) $(CUTOS)
	$(CUC) $(CUFLAGS) $(CUOPT) -o $@ $^ $(CULIBS)

cutestcvtd: testcvtcuda.cu threshdebug.o threshcudadebug.o $(CUOSDEB) $(CUWOSDEB) $(OSDEB) $(WOSDEB) $(TOSDEB) $(CUTOSDEB)
	$(CUC) $(CUFLAGS) $(CUG) -o $@ $^ $(CULIBS)

cptestcvt: testcvt.cpp thresh.o $(OS) $(WOS) $(TOS)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LIBS)

cptestcvtd: testcvt.cpp threshdebug.o $(OSDEB) $(WOSDEB) $(TOSDEB)
	$(CC) $(CFLAGS) $(G) -o $@ $^ $(LIBS)

cptesttransform: testtransform.cpp $(OS) $(WOS) $(TOS)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LIBS)

cptesttransformd: testtransform.cpp $(OSDEB) $(WOSDEB) $(TOSDEB)
	$(CC) $(CFLAGS) $(G) -o $@ $^ $(LIBS)

%cuda.o: %cuda.cu $(CUOS) %coeffs.h %cuda.cuh
	$(CUC) $(CUFLAGS) $(CUOPT) $(CULINK) -c -o $@ $< $(CULIBS)

%cudadebug.o: %cuda.cu $(CUOSDEB) %coeffs.h %cuda.cuh
	$(CUC) $(CUFLAGS) $(CUG) $(CULINK) -c -o $@ $< $(CULIBS)

%.o: %.cpp %.h %coeffs.h $(OS)
	$(CC) $(CFLAGS) $(OPT) -c -o $@ $< $(LIBS)

%debug.o: %.cpp %.h %coeffs.h $(OSDEB)
	$(CC) $(CFLAGS) $(G) -c -o $@ $< $(LIBS)

transform.o: transform.cpp $(OS) $(filter-out thresh.o,$(WOS))
	$(CC) $(CFLAGS) $(OPT) -c -o $@ $< $(LIBS)

transformdebug.o: transform.cpp $(OSDEB) $(filter-out threshdebug.o,$(WOSDEB))
	$(CC) $(CFLAGS) $(G) -c -o $@ $< $(LIBS)

transformcuda.o: transformcuda.cu $(CUOS) $(filter-out threshcuda.o,$(CUWOS))
	$(CUC) $(CUFLAGS) $(CUOPT) -c -o $@ $< $(LIBS)

transformcudadebug.o: transformcuda.cu $(CUOSDEB) $(filter-out threshcudadebug.o,$(CUWOSDEB))
	$(CUC) $(CUFLAGS) $(CUG) -c -o $@ $< $(LIBS)

thresh.o: thresh.cpp $(OS) $(TOS)
	$(CC) $(CFLAGS) $(OPT) -c -o $@ $< $(LIBS)

threshdebug.o: thresh.cpp $(OSDEB) $(TOSDEB)
	$(CC) $(CFLAGS) $(G) -c -o $@ $< $(LIBS)

threshcuda.o: threshcuda.cu $(CUOS) $(CUTOS)
	$(CUC) $(CUFLAGS) $(CUOPT) $(CULINK) -c -o $@ $< $(CULIBS)

threshcudadebug.o: threshcuda.cu $(CUOSDEB) $(CUTOSDEB)
	$(CUC) $(CUFLAGS) $(CUG) $(CULINK) -c -o $@ $< $(CULIBS)

utils.o: utils.cpp $(HEADS)
	$(CC) $(CFLAGS) $(OPT) -c -o $@ $< $(LIBS)

utilsdebug.o: utils.cpp $(HEADS)
	$(CC) $(CFLAGS) $(G) -c -o $@ $< $(LIBS)

utilscuda.o: utilscuda.cu $(HEADS) $(CUHEADS)
	$(CUC) $(CUFLAGS) $(CUOPT) $(CULINK) -c -o $@ $< $(CULIBS)

utilscudadebug.o: utilscuda.cu $(HEADS) $(CUHEADS)
	$(CUC) $(CUFLAGS) $(CUG) $(CULINK) -c -o $@ $< $(CULIBS)

wavutils.o: wavutils.cpp $(HEADS)
	$(CC) $(CFLAGS) $(OPT) -c -o $@ $< $(LIBS)

wavutilsdebug.o: wavutils.cpp $(HEADS)
	$(CC) $(CFLAGS) $(G) -c -o $@ $< $(LIBS)

wavutilscuda.o: wavutilscuda.cu cudaheads.h
	$(CUC) $(CUFLAGS) $(CUOPT) $(CULINK) -c -o $@ $< $(CULIBS)

wavutilscudadebug.o: wavutilscuda.cu $(CUHEADS)
	$(CUC) $(CUFLAGS) $(CUG) $(CULINK) -c -o $@ $< $(CULIBS)

.PHONY: clean
clean:
	rm -f *.o *~ $(EXES) $(CUEXES) $(PROF) $(EXESDEB) $(CUEXESDEB)
