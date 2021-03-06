#ifndef RFUNC_H
#define RFUNC_H

#include<R.h>
#include<Rinternals.h>
#include <R_ext/Rdynload.h>

#include "transform.h"
#include "transformcuda.cuh"
#include "thresh.h"
#include "threshcuda.cuh"

#define MANUAL 0
#define UNIV 1
#define CV 2

extern "C" {
  void RcpuTransform(real* x, real* xmod, int *len, int * sense, int* nlevels, int * ttype, int * filter, int * filterlen);
  // transform a given vector input using the CPU
  // x and xmod assumed already allocation (if needed)

  void RgpuTransform(real* x_h, real* xmod_h, int *len, int * sense, int* nlevels, int * ttype, int * filter, int * filterlen);
  // transform a given vector input using the GPU
  // x and xmod assumed already allocation (if needed)
  

  void RcpuThreshold(real* x, real* xmod, int* len, int* nlevels, int * ttype, int* filter, int * filterlen, real* thresh, int* hardness, int* minlevel, int*maxlevel);
  // threshold a DWT or MODWT on the CPU


  void RcpuSmooth(real* x, int* len, int* nlevels, int * ttype, int* filter, int * filterlen, int* threshtype, real* thresh, int* hardness, int* minlevel, int* maxlevel, real* tol);
  // FWD transform, threshold & BWD transform using the CPU
  

  void RgpuSmooth(real* x, int* len, int* nlevels, int * ttype, int* filter, int * filterlen, int* threshtype, real* thresh, int* hardness, int* minlevel, int* maxlevel, real* tol);
    // FWD transform, threshold & BWD transform using the GPU

SEXP RgpuTransformList(SEXP argslist);

}

#endif