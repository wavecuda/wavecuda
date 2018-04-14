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

#endif //ifndef