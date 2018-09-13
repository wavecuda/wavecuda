#ifndef C6CUDA_CUH
#define C6CUDA_CUH

#include "utils.h"
#include "utilscuda.cuh"
#include "wvtheads.h"
#include "cudaheads.h"
#include "wavutilscuda.cuh"

int C6CUDA_sh(real* x_d, uint len, short int sense, uint nlevels);
// using 1 kernel, shared memory, uses vector of boundary points
int fC6CUDAsh(real* x_d, uint len, uint skip, uint nlevels);
int bC6CUDAsh(real* x_d, uint len, uint skip);

__global__ void C6_kernel_shared_f(real* x, const uint len, const uint skip, real*, const uint);
__global__ void C6_kernel_shared_b(real* x, const uint len, const uint skip, real*, const uint);

__device__ double get_wvt_shared(real* x, real* bdrs, const uint len, const uint skip, const int i, const int ish, const uint tid, const uint bid, const short isskip);

int C6CUDA_sh_ml2(real* x_d, uint len, short int sense, uint nlevels);
int fC6CUDAsh_ml2(real* x_d, uint len, uint skip, uint nlevels);
int bC6CUDAsh_ml2(real* x_d, uint len, uint skip);

__global__ void C6_kernel_shared_f_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb);
__global__ void C6_kernel_shared_b_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb);




static __device__ void lift_cyc_1(real* xsh, const int ish, const uint sk, const short int sense);
static __device__ void lift_cyc_2(real* xsh, const int ish, const uint sk, const short int sense);
static __device__ void lift_cyc_3(real* xsh, const int ish, const uint sk, const short int sense);
static __device__ void lift_cyc_4(real* xsh, const int ish, const uint sk, const short int sense);
static __device__ void lift_cyc_5(real* xsh, const int ish, const uint sk, const short int sense);
static __device__ void lift_cyc_6(real* xsh, const int ish, const uint sk, const short int sense);


int C6CUDA_sh_ml2_streams(real* x_h, real* x_d, uint len, short int sense, uint nlevels, cudaStream_t stream);

#endif //ifndef