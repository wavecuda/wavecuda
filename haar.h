#ifndef HAAR_H
#define HAAR_H

#include "wvtheads.h"

int Haar(real* x, uint len, short int sense, uint nlevels);
int fHaar(real* x, uint len, uint skip, uint nlevels);
int bHaar(real* x, uint len, uint skip);

int Haarmp(real* x, uint len, short int sense, uint nlevels);
int fmpHaar(real* x, uint len, uint skip, uint nlevels);
int bmpHaar(real* x, uint len, uint skip);

int HaarCoalA(real* x, uint len, short int sense);
int fHaarCA(real* x, uint len, uint pos);
int bHaarCA(real* x, uint len, uint pos);

int HaarMODWT(real* x, real* xdat, uint len, short int sense, uint nlevels);
int fHaarMODWT(real* x, real* xdat, uint len, uint skip, uint nlevels);
int bHaarMODWT(real* x, real* xdat, uint len, uint skip, uint nlevels);

// omp1 uses the omp Haar function
// (NB, this is very slow!)
int HaarMODWTomp1(real* x, real* xdat, uint len, short int sense, uint nlevels);
int fHaarMODWTomp1(real* x, real* xdat, uint len, uint skip, uint nlevels);

// omp2 runs through the shifts in parallel
// this is faster than the original, serial implementation
int HaarMODWTomp2(real* x, real* xdat, uint len, short int sense, uint nlevels);
int fHaarMODWTomp2(real* x, real* xdat, uint len, uint skip, uint nlevels);
int bHaarMODWTomp2(real* x, real* xdat, uint len, uint skip, uint nlevels);

// Haar function implementation that isn't carried out in-place
// so it writes to separate scaling & detail vectors
// in practice, this means it doesn't need a temporary variable for storage
// & it's immediately parallelisable
// and the different skip vals mean we can control whether/how to do decimation
int fHaarSDout(real* Xin, real* Sout, real* Dout, uint len, uint incri, uint skipi, uint skipo, uint nlevels);

// This is a time-ordered implementation (as supposed to packet-ordered)
// this doesn't involve copying of data for in-place, packet-ordered transforms
// instead, we write directly to the xdat vector
int HaarMODWTto(real* x, real* xdat, uint len, short int sense, uint nlevels);
int fHaarMODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels);
int bHaarMODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels);

// This is an omp version of the time-ordered implementation above
int fHaarSDoutomp(real* Xin, real* Sout, real* Dout, uint len, uint incri, uint skipi, uint skipo, uint nlevels);
int bHaarSDoutomp(real* Xout, real* Sin, real* Din, uint len, uint incri, uint skipi, uint skipo, uint nlevels);
int HaarMODWTtomp(real* x, real* xdat, uint len, short int sense, uint nlevels);
int fHaarMODWTtomp(real* x, real* xdat, uint len, uint skip, uint nlevels);
int bHaarMODWTtomp(real* x, real* xdat, uint len, uint skip, uint nlevels);

#endif //ifndef
