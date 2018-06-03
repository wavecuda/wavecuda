#ifndef WVTHEADS_H
#define WVTHEADS_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>
#include <math.h>

#define real double

typedef unsigned int uint;

#define HARD 0
#define SOFT 1

#define DOUBLESIZE 8
#define FLOATSIZE 4

#define FWD 1
#define BWD 0

#define HAAR 1
#define DAUB4 2
#define C6F 3
#define LA8F 4
#define HAARMP -1
#define DAUB4MP -2
#define C6FMP -3
#define LA8FMP -4
#define HAARNOHOST -1 // GPU transform using only device, no host memory
#define DAUB4NOHOST -2 // assumed in code to be (-1) * normalfiltercode
#define C6NOHOST -3
#define LA8NOHOST -4

#define DWT 0
#define MODWT_TO 1
#define MODWT_PO 2

typedef struct wvtstruct{
  // wavelet structure for CPU implementations
  
  real* x; // DWT or orig vector if trans type is MODWT
  short ttype; // type of trans: DWT, TO MODWT or PO MODWT
  short filt; // the wavelet filter: Haar, D4 etc
  short filtlen; // filter length
  short transformed; // transformed to wavelet domain?
  uint levels; // level of transform. 0 => no transform
  uint len; // length of x
  real* xmod;  // NULL if DWT, transformed vector if MODWT
} wst;

typedef struct cuwvtstruct{
  // wavelet structure for GPU implementations
  
  real* x_h; // DWT or orig vector if trans type is MODWT (host)
  real* x_d; // (device)
  short ttype; // type of trans: DWT, TO MODWT or PO MODWT
  short filt; // the wavelet filter: Haar, D4 etc
  short filtlen; // filter length
  short transformed; // transformed to wavelet domain?
  uint levels; // level of transform. 0 => no transform
  uint len; // length of x
  real* xmod_h;  // NULL if DWT, transformed vector if MODWT (host)
  real* xmod_d;  // (device)
} cuwst;


#include "wavutils.h" // this include is put after the above definitions, as they are required.

#endif //ifndef
