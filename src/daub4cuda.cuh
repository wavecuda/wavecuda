#ifndef DAUB4CUDA_CUH
#define DAUB4CUDA_CUH

#include "utils.h"
#include "utilscuda.cuh"
#include "wvtheads.h"
#include "cudaheads.h"
#include "wavutilscuda.cuh"

int Daub4CUDA(real* x, uint len, short int sense, uint nlevels);
// version using 4 kernels, no shared memory
int fDaub4CUDA(real* x, uint len, uint skip, uint nlevels);
int bDaub4CUDA(real* x, uint len, uint skip); //doesn't make sense to give backward transform the number of levels, because we want the original coeffs back in our inverse transform. Well. Also, our 'skip' variable is a function of nlevels, so nlevels goes in there. Transforming back to a different level would require another input. & it could easily be done by another forward transformation.

__global__ void Daub4_kernel1(real* x, const uint len, const uint skip, short int sense);
__global__ void Daub4_kernel2(real* x, const uint len, const uint skip, short int sense);
__global__ void Daub4_kernel3(real* x, const uint len, const uint skip, short int sense);
__global__ void Daub4_kernel4(real* x, const uint len, const uint skip, short int sense);

int Daub4CUDA_sh(real* x_d, uint len, short int sense, uint nlevels);
// using 1 kernel, shared memory, uses vector of boundary points
int fDaub4CUDAsh(real* x_d, uint len, uint skip, uint nlevels);
int bDaub4CUDAsh(real* x_d, uint len, uint skip);

// daub4_sh specific get_bdrs function
__global__ void get_bdrs_sh(real* x, const uint len, const uint skip, real* bdrs, const uint lenb);

// __global__ void Daub4_kernel_shared_f(real* x, const uint len, const uint skip, real*, real*, const uint);

__global__ void Daub4_kernel_shared_f(real* x, const uint len, const uint skip, real*, const uint);


__global__ void Daub4_kernel_shared_b(real* x, const uint len, const uint skip, real*, const uint);

int Daub4CUDA_sh_ml2(real* x_d, uint len, short int sense, uint nlevels);
// 1 kernel, multi-level, shared memory, uses vector of boundary points
int fDaub4CUDAsh_ml2(real* x_d, uint len, uint skip, uint nlevels);
int bDaub4CUDAsh_ml2(real* x_d, uint len, uint skip);


__global__ void Daub4_kernel_shared_f_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb);

__global__ void Daub4_kernel_shared_b_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb);

int Daub4CUDA_sh_io(real* x_d_in, real* x_d_out, uint len, short int sense, uint nlevels);
// 1 kernel, in/output vectors (avoiding boundary vector), shared memory
int fDaub4CUDAsh_io(real* x_d_in, real* x_d_out, uint len, uint skip, uint nlevels);
int bDaub4CUDAsh_io(real* x_d_in, real* x_d_out, uint len, uint skip);

__global__ void Daub4_kernel_shared_f_io(real* x_in, real* x_out, const uint len, const uint skip);

__global__ void Daub4_kernel_shared_b_io(real* x_in, real* x_out, const uint len, const uint skip);

int Daub4CUDA_sh_ml2_io(real* x_d_in, real* x_d_out, uint len, short int sense, uint nlevels);
// 1 kernel, in/output vectors (avoiding boundary vector), shared memory, multi-level
int fDaub4CUDAsh_ml2_io(real* x_d_in, real* x_d_out, uint len, uint skip, uint nlevels);
int bDaub4CUDAsh_ml2_io(real* x_d_in, real* x_d_out, uint len, uint skip);

__global__ void Daub4_kernel_shared_f_ml2_io(real* x_in, real* x_out, const uint len, const uint skip);

__global__ void Daub4_kernel_shared_b_ml2_io(real* x_in, real* x_out, const uint len, const uint skip);


#endif //ifndef