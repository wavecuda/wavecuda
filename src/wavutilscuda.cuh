#ifndef WAVUTILS_CUH
#define WAVUTILS_CUH

#include "cudaheads.h"
#include "utilscuda.cuh"
#include "wavutils.h"

#define BS_BD 1024 // block size for the boundary setting kernel

// general get bdrs function for CUDA wavelet kernels
// taking k extra coefficients each side
__global__ void get_bdrs_sh_k(real* x, const uint len, const uint skip, real* bdrs, const uint lenb, const uint k, const uint block_sect);

// general shared memory filling function
// returning a extra coefficients at lower end of shared vec
// & b extra coeffs at upper end of shared vec
__device__ double get_wvt_shared_gen(real* x, real* bdrs, const uint len, const uint skip, const int i, const int ish, const uint a, const uint b, const uint bsect, const short isskip);

// same as above, except we write directly to the memory!
__device__ void write_wvt_shared_gen(real* x, real* bdrs, const uint len, const uint skip, const int i, const int ish, const uint a, const uint b, const uint bsect, real * x_sh);

cuwst* create_cuwvtstruct(short ttype, short filt, uint filtlen, uint levels, uint len);
// default create cuwvtstruct : creat struct with host memory
// wrapper to function below

cuwst* create_cuwvtstruct(short ttype, short filt, uint filtlen, uint levels, uint len, short hostalloc);
// create cuwvtstruct with option of allocating host memory
// eg in CVT, we do all work on the GPU to avoid memory transfers between host & device

void kill_cuwvtstruct(cuwst *w);
void kill_cuwvtstruct(cuwst *w, short hostallocated);


cuwst* dup_cuwvtstruct(cuwst *w1, short memcpy, short hostalloc);
// duplicate a cuwst object, with an option of not
// copying memory, in case that is ever required

cuwst* dup_cuwvtstruct(cuwst *w1);
// duplicating a cuwst object with memcopy & host allocation

wst* cpu_alias_cuwvtstruct(cuwst* w_gpu);
// create a wst object alias from a cuwst object
// that shares the same memory as the cuwst object
// this allows us to use CPU functions
// at the expense of allocating a few integers

void kill_alias_wvtstruct(wst *w);
// free the structure but not the arrays inside

void print_cuwst_info(cuwst *w);
// print details about cuwst type

void update_cuwst_host(cuwst *w);
  // function for doing appropriate cudamemcpy ops
  // written for debugging  
  // if modwt, then we update either xmod or x,
  // depending on whether w is transformed
  // isn't clever enough to just update recently thresholded levels of detail!

void update_cuwst_device(cuwst *w);
  // function for doing appropriate cudamemcpy ops
  // written for debugging  
  // if modwt, then we update either xmod or x,
  // depending on whether w is transformed
  // isn't clever enough to just update recently thresholded levels of detail!

uint ndetail_thresh(cuwst* w, uint minlevel, uint maxlevel);
// wrapper to function in wavutils but taking nice arguments
// returns the number of detail coefficients inside thresholding levels

#endif //ifndef