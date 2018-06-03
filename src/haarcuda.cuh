#ifndef HAAR_CUDA_CUH
#define HAAR_CUDA_CUH

#include "utils.h"
#include "utilscuda.cuh"
#include "wvtheads.h"
#include "cudaheads.h"
#include "wavutils.h"

// plain, slow CUDA Haar transform without shared mem
int HaarCUDA(real* x_d, uint len, short int sense, uint nlevels);
int fHaarCUDA(real* x_d, uint len, uint skip, uint nlevels);
int bHaarCUDA(real* x_d, uint len, uint skip);

// Basic CUDA Haar transform with shared mem
int HaarCUDAsh(real* x_d, uint len, short int sense, uint nlevels);
int fHaarCUDAsh(real* x_d, uint len, uint skip, uint nlevels);
int bHaarCUDAsh(real* x_d, uint len, uint skip);

__global__ void Haar_kernel(real* x, const uint len, const uint skip);
__global__ void Haar_kernel_vars(real* x, const uint len, const uint skip);
__global__ void Haar_kernel_shared(real* x, const uint len, const uint skip);
__global__ void Haar_kernel_shared_ml_f(real* x, const uint len, const uint skip, const uint levels);
__global__ void Haar_kernel_shared_ml_b(real* x, const uint len, const uint skip, const uint levels);

// CUDA Haar transform using no shared mem but coalesced global mem
int HaarCUDACoalA(real* x_d, uint len, short int sense);
int fHaarCUDACA(real* x_d, uint len, uint pos);
int bHaarCUDACA(real* x_d, uint len, uint skip);
__global__ void Haar_CA_kernelf(real* x, const uint len, const uint pos, real* s, const uint ssize);
__global__ void Coal_copy_kernelf(real* x, const uint pos, real* s, const uint ssize);

// multi-level Haar CUDA transform
int HaarCUDAML(real* x_d, uint len, short int sense, uint nlevels);
int fHaarCUDAML(real* x_d, uint len, uint skip, uint nlevels);
int bHaarCUDAML(real* x_d, uint len, uint skip);

// multi-level Haar CUDA transform
// complete with stream & memcpy
int HaarCUDAMLv2(real* x_h, real* x_d, uint len, short int sense, uint nlevels, cudaStream_t stream);
int fHaarCUDAMLv2(real* x_h, real* x_d, uint len, uint skip, uint nlevels, cudaStream_t stream);
int bHaarCUDAMLv2(real* x_h, real* x_d, uint len, uint skip, cudaStream_t stream);

// multi-level Haar CUDA transform with stream
// all with device memory, no memcpy
int HaarCUDAMLv3(real* x_d, uint len, short int sense, uint nlevels, cudaStream_t stream);
int fHaarCUDAMLv3(real* x_d, uint len, uint skip, uint nlevels, cudaStream_t stream);
int bHaarCUDAMLv3(real* x_d, uint len, uint skip, cudaStream_t stream);

// packet ordered MODWT, using device memory, no CPU/GPU memcopy. Slow!
int HaarCUDAMODWT(real* x_d, real* xdat_d, uint len, short int sense, uint nlevels);
int fHaarCUDAMODWT(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels);
__global__ void Haar_kernel_MODWT(real* x_in, real* x_out, const uint len, const uint skip, const uint shift);
int bHaarCUDAMODWT(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels);
__global__ void Haar_kernel_MODWT(real* x_in, real* x_out, const uint len, const uint skip, const uint shift);


// po MODWT, using device memory. Also using streams (probably badly!) Also slow!
int HaarCUDAMODWTv2(real* x_d, real* xdat_d, uint len, short int sense, uint nlevels);
int fHaarCUDAMODWTv2(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels);

// po MODWT, using host mem input, streams, async transfers, also slow!
int HaarCUDAMODWTv3(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, short int sense, uint nlevels);
int fHaarCUDAMODWTv3(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, uint skip, uint nlevels);

// po MODWT, host mem input, streams, async transfers, 2 shifts per kernel, still slow! (but slightly faster)
int HaarCUDAMODWTv4(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, short int sense, uint nlevels);
int fHaarCUDAMODWTv4(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, uint skip, uint nlevels);

__global__ void Haar_kernel_MODWT_v2(real* x_in, real* x_out, const uint len, const uint skip);

// to MODWT, device mem throughout
int HaarCUDAMODWTv5(real* x_d, real* xdat_d, uint len, short int sense, uint nlevels);
int fHaarCUDAMODWTv5(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels);
int bHaarCUDAMODWTv5(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels);

__global__ void Haar_kernel_MODWT_v3(real* x_in, real* s_out, real* d_out, const uint len, const uint skip);

__global__ void Haar_kernel_MODWT_v3_1(real* x_in, real* s_out, real* d_out, const uint len, const uint skip);

__global__ void Haar_kernel_MODWT_v3_b(const real* s_in, const real* d_in, real* x_out, const uint len, const uint skip);

__global__ void Haar_kernel_MODWT_v3_bsh(const real* s_in, const real* d_in, real* x_out, const uint len, const uint skip);

// to MODWT, has a stream pointer passed to it, does async mem copy
// fastest!
int HaarCUDAMODWTv6(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, short int sense, uint nlevels, cudaStream_t stream);
int fHaarCUDAMODWTv6(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, uint skip, uint nlevels, cudaStream_t stream);
int bHaarCUDAMODWTv6(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, uint skip, uint nlevels, cudaStream_t stream);

// to MODWT, takes a stream pointer, all device memory
// so faster than above but needs to be used by routines that don't want host memory
int HaarCUDAMODWTv6d(real* x_d, real* xdat_d, uint len, short int sense, uint nlevels, cudaStream_t stream);
int fHaarCUDAMODWTv6d(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels, cudaStream_t stream);
int bHaarCUDAMODWTv6d(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels, cudaStream_t stream);


#endif