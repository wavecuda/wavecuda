#ifndef UTILS_CU_H
#define UTILS_CU_H

#include "utils.h"
#include "wvtheads.h"
#include "cudaheads.h"

__global__ void copyveccu(const real* from_d, real* to_d, uint len);

__global__ void uint2realcu(const uint* from_uint_d, real* to_real_d, uint len);

void initrandveccu(real* x, uint len);

__global__ void cmpveccu(real* v1_d, real* v2_d, uint len);

__global__ void printveccu(real* x_d, uint len);

__global__ void printmatveccu(real* x_d, uint nrow, uint ncol);

void print_device_vector(real* x_d, uint len);

#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val);
// from cuda programming guide
// manual double atomic add as not possible in compute 3/3.5

// we should be able to overload atomicAdd, but for some reason this isn't working
// so instead we create a new function
#endif


#endif //ifndef