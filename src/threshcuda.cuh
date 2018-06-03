#ifndef THRESHCUDA_H
#define THRESHCUDA_H

#include "wvtheads.h"
#include "thresh.h"
#include "cudaheads.h"
#include "wavutilscuda.cuh"
#include "transformcuda.cuh"

//#define mtype float
#define mtype real

__global__ void thresh_dwt_cuda_kernel_hard(real* in, real* out, const real thresh, const uint len, const uint minlevel, const uint maxlevel);

__global__ void thresh_dwt_cuda_kernel_soft(real* in, real* out, const real thresh, const uint len, const uint minlevel, const uint maxlevel);

__global__ void thresh_modwt_cuda_kernel(real* in, real* out, const real thresh, const uint len, short hardness, const short modwttype, const uint minlevel, const uint maxlevel, const uint levels);

__device__ real thresh_coef_cuda_hard(const real coef_in, const real thresh);

__device__ real thresh_coef_cuda_soft(const real coef_in, const real thresh);


__device__ double atomicAdd(double* address, double val);
// from cuda programming guide
// manual double atomic add as not possible in compute 3/3.5

void threshold(cuwst* win, cuwst* wout, real thresh, short hardness, uint minlevel, uint maxlevel, cudaStream_t stream);

real interp_mse(cuwst* wn, cuwst* wye, cuwst* wyo, mtype *m_d, cudaStream_t stream);
// CUDA verson of interp mse function

__global__ void interp_mse_kernel(real* xn, real* ye, real* yo, uint len, mtype* m_global);
// CUDA kernel that does the interp_mse calculation

real CVT(cuwst *w, short hardness, real tol, uint minlevel, uint maxlevel);
// CUDA version of cross validation!
// takes a gpu wavelet object, untransformed
// returns a threshold, modifies object to be thresholded


__device__ short is_in_d_level_limits(uint i, uint len, uint minlevel, uint maxlevel);
// short function to ascertain whether an index i of a DWT vector is
// a detail coefficient inside the limits (for thresholding)

__global__ void sum_n_sqdev_dwt_details(real* x_d, const uint len, real *sum_d, const real *m_d, const short ttype, const uint minlevel, const uint maxlevel, uint n_det, const short sqdev);
// calculates
// (1/n_det) * sum [ d_i - m ]
// or
// (1/(n_det-1)) * sum [ (d_i - m)^2 ]
// for DWT detail coeffs d_i in thresholding levels, n_det in number

__global__ void sum_n_sqdev_modwt_details(real* x_d, const uint len, real *sum, const real *m_d, const short ttype, const uint minlevel, const uint maxlevel, uint n_det, const short sqdev);
// calculates
// (1/n_det) * sum [ d_i - m ]
// or
// (1/(n_det-1)) * sum [ (d_i - m)^2 ]
// for MODWT detail coeffs d_i in thresholding levels, n_det in number

__device__ void sum_reduce_shmem(real* xsh, uint skip, uint len, uint sh_size, uint i, uint ish);
// puts the sum of shared memory block xsh into xsh[0]
// via sum reduce
// assumes len, skip are powers of 2

real univ_thresh_approx(cuwst *w, uint minlevel, uint maxlevel, cudaStream_t stream);


#endif