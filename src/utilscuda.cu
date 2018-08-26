#include "utilscuda.cuh"

// Copy a vector of length 'len' into another vector
// Both vectors must already be allocated
__global__ void copyveccu(const real* from_d, real* to_d, uint len){
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  // for(i=0;i<len;i++) to[i]=from[i];
  if(i < len) to_d[i] = from_d[i];
}

__global__ void uint2realcu(const uint* from_uint_d, real* to_real_d, uint len){
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) to_real_d[i] = (real) from_uint_d[i];
}

// Puts random integers into the first 'len' elements of an allocated vector on the device, declared as type real*
void initrandveccu(real* x_d, uint len){
  // inefficient, but just for example purposes
  // for(i=0;i<len;i++) x[i]=rand();
  uint *randints;
  cudaError_t cuderr;
  cudaMalloc((void **)&randints,len*sizeof(uint));
  curandGenerator_t gen;
  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT );
  curandGenerate(gen , randints , len);
  int threadsPerBlock = 256;
  int blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
  uint2realcu<<<blocksPerGrid, threadsPerBlock>>>(randints,x_d,len);
  cuderr = cudaGetLastError();
  if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in uint2real kernel (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
  cuderr = cudaFree(randints);
  if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in cuda free (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
}

// Compares the first 'len' elements of two vectors
// prints indexes where they differ
// returns #elts where they differ
__global__ void cmpveccu(real* v1_d, real* v2_d, uint len){
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ uint errs;
  // for(i=0;i<len;i++){
  //   if((v1[i]<v2[i]*(1-FLT_EPSILON)) && (v1[i]>v2[i]*(1+FLT_EPSILON))){
  //     printf("Vectors differ at index %i/n",i);
  //     errs+=1;
  //   }
  // }
  // if(errs==0) printf("Congrats: no errors!\n");
  if (i < len){
    if((v1_d[i]<v2_d[i]*(1-FLT_EPSILON)) || (v1_d[i]>v2_d[i]*(1+FLT_EPSILON))){
      printf("Vectors differ at index %i/n",i);
      atomicAdd(&errs,1);
    }
  }
  __syncthreads();
  if (i==1){
    printf("\ncmpveccu: %i errors found\n",errs);
  }
}

__global__ void printveccu(real* x_d, uint len){
  uint i;
  printf("\nPrint vector on device...\n");
  for(i=0;i<len;i++) printf("%g\n",x_d[i]);
  printf("\n");
}

__global__ void printmatveccu(real* x_d, uint nrow, uint ncol){
  //this is a 1D matrix array
  uint i,j;
  printf("\nPrint matrix on device...\n");
  for(i=0;i<nrow*ncol;i+=ncol){
    for(j=0;j<ncol;j++){
      printf("%g,",x_d[i+j]);
    }
    printf("\n");
  }
  printf("\n");
}

void print_device_vector(real* x_d, uint len){
  // debugging bodge function to print stuff on device memory
  // without having to wait for a kernel
  real tmp[len];
  cudaStream_t str;
  cudaStreamCreate(&str);
  cudaMemcpyAsync(tmp,x_d,len*sizeof(real),DTH,str);
  cudaStreamSynchronize(str);
  printvec(tmp,len);
  cudaStreamDestroy(str);
}


#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val) {
  // from cuda programming guide
  // manual double atomic add as not possible in compute 3/3.5
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(val +
					 __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
