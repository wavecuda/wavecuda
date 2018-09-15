#ifndef TRANSCUDA_CUH
#define TRANSCUDA_CUH

#include "wvtheads.h"
#include "daub4cuda.cuh"
#include "haarcuda.cuh"
#include "c6cuda.cuh"
#include "la8cuda.cuh"

int transform(cuwst *w,short sense);
// when we don't want to run with streams
// will be run with NULL stream, i.e. the default stream

int transform(cuwst *w,short sense, cudaStream_t stream);
// for transforming with specified stream


#endif