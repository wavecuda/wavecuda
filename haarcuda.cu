#include "haarcuda.cuh"
#include "haarcoeffs.h"

#define BLOCK_SIZE 256
//#define BLOCK_SIZE 128

/*---------------------------------------------------------
  Functions for basic GPU (CUDA) wavelet transform
  ---------------------------------------------------------*/


/*
  data structure used is the following:
  x is our vector

  x                        transform
  i 
  0   | 0 |   |s11|   |s21|   |s31|
  1   | 0 |   |d11|   |d11|   |d11|
  2   | 5 |   |s12|   |d21|   |d21|
  3   | 4 |   |d12|   |d12|   |d12|
  4   | 8 |   |s13|   |s22|   |d31|
  5   | 6 |   |d13|   |d13|   |d13|
  6   | 7 |   |s14|   |d22|   |d22|
  7   | 3 |   |d14|   |d14|   |d14|

  skip=1  skip=2  skip=4  skip=8

  ----> forward transform ---->
  <--- backward transform <----

  where sij (dij) is the scaling (detail) coefficient number j at level i.
*/

int HaarCUDA(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDA(x_d,len,1,nlevels));
  case BWD:
    return(bHaarCUDA(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaarCUDA(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;

    Haar_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip);
    
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fHaar (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    return(fHaarCUDA(x_d,len,skip<<1,nlevels));

  }
  return(0);
}

int bHaarCUDA(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;
    Haar_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bHaar (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    return(bHaarCUDA(x_d,len,skip>>1));
  }
  return(0);
}

__global__ void Haar_kernel(real* x, const uint len, const uint skip){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip<<1;
  real tmp;

  if (i < len){
    tmp = (x[i] - x[i+skip])*invR2;
    x[i] = (x[i] + x[i+skip])*invR2;
    x[i+skip] = tmp;
  }
}

__global__ void Haar_kernel_vars(real* x, const uint len, const uint skip){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip<<1;
  real xi, xi1;
  if (i < len){
    xi = x[i];
    xi1 = x[i+skip];
    
    x[i] = (xi + xi1)*skip;
    x[i+skip] = (xi - xi1)*skip;
  }
}

int HaarCUDAsh(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDAsh(x_d,len,1,nlevels));
  case BWD:
    return(bHaarCUDAsh(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaarCUDAsh(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;

    Haar_kernel_shared<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip);
    
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fHaar (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    return(fHaarCUDAsh(x_d,len,skip<<1,nlevels));

  }
  return(0);
}

int bHaarCUDAsh(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;
    Haar_kernel_shared<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bHaar (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    return(bHaarCUDAsh(x_d,len,skip>>1));
  }
  return(0);
}



/*---------------------------------------------------------
  First attempt at shared memory - within-level
  ---------------------------------------------------------*/

// 48KB of shared memory available per block.
// blocksize = 64 => storing 128 doubles will use 128 * 8B = 1024B = 1KB
// loads of space!
__global__ void Haar_kernel_shared(real* x, const uint len, const uint skip){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip<<1;
  // counter inside vector x
  uint ish = threadIdx.x<<1;
  // counter inside shared vector x_work

  __shared__ real x_work[2*BLOCK_SIZE];
  
  if (i < len){
    x_work[ish] = x[i];
    x_work[ish+1] = x[i + skip];
  }
  
  __syncthreads();

  if (i < len){
    x[i] = (x_work[ish] + x_work[ish+1])*invR2;
    x[i+skip] = (x_work[ish] - x_work[ish+1])*invR2;
  }
  
}

/*---------------------------------------------------------
  Functions for GPU (CUDA) wavelet transform with Multi-level kernels
  ---------------------------------------------------------*/

int HaarCUDAML(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case 1:
    return(fHaarCUDAML(x_d,len,1,nlevels));
  case 0:
    return(bHaarCUDAML(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaarCUDAML(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/(skip<<1) + threadsPerBlock - 1) / threadsPerBlock;
    
    uint levels=1; //leave at 1. This is initialisation for level variable!

    //printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    
    while((levels+1<=2)&&((skip<<(levels+1))<=(1 << nlevels))){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }
    Haar_kernel_shared_ml_f<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,levels);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fHaar ML (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    //printf("CUDA: len=%u,skip=%u\n",len,skip);
    //printveccu<<<1,1>>>(x_d,len);
    cudaDeviceSynchronize();

    return(fHaarCUDAML(x_d,len,skip<<levels,nlevels));

  }
  return(0);
}

int bHaarCUDAML(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;

    uint levels=1; //leave at 1. This is initialisation for level variable!

    //printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    
    while((levels+1<=2)&&((skip>>levels)>0)){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }
    Haar_kernel_shared_ml_b<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,levels);

    //Haar_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bHaar ML (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    return(bHaarCUDAML(x_d,len,skip>>levels));
  }
  return(0);
}



/*---------------------------------------------------------
  Second attempt at shared memory - multi-level
  ---------------------------------------------------------*/

__global__ void Haar_kernel_shared_ml_f(real* x, const uint len, const uint skip, const uint levels){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip<<1;
  // counter inside vector x
  uint ish = threadIdx.x<<1;
  // counter inside shared vector x_work
  uint li; // counter representing the number of levels
  real tmp;
  __shared__ real x_work[BLOCK_SIZE<<1];
  uint skipwork = 1;

  // copy into shared memory
  if (i < len){
    x_work[ish] = x[i];
    x_work[ish+1] = x[i + skip];
  }
  
  __syncthreads();

  //then we loop over a few levels

  for(li = 0; li < levels; li++){

    if (ish < (BLOCK_SIZE<<1)){
      tmp = (x_work[ish] - x_work[ish+skipwork])*invR2;
      x_work[ish] = (x_work[ish] + x_work[ish+skipwork])*invR2;
      x_work[ish+skipwork] = tmp;
    }
    
    ish=ish<<1; skipwork=skipwork<<1;
    __syncthreads();
  }
  
  ish = threadIdx.x<<1;
  
  if (i < len){
    x[i] = x_work[ish];
    x[i + skip] = x_work[ish+1];
  }
  
}

//backwards
__global__ void Haar_kernel_shared_ml_b(real* x, const uint len, const uint skip, const uint levels){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip<<1>>(levels-1);
  // counter inside vector x
  uint ish = threadIdx.x<<1; // we need this for copying vec - later, we need wider ishes
  // counter inside shared vector x_work
  uint wsize = min(len/(skip>>(levels-1)),BLOCK_SIZE<<1); //size of shared vec x_work
  uint li; // counter representing the number of levels
  real tmp;
  __shared__ real x_work[BLOCK_SIZE<<1];//shared mem must be of size 'const'
  // we only use wsize elements of x_work
  uint skipwork = 1<<(levels-1);
  
  // if(threadIdx.x == 1){
  //   printf("\n###Backward trans###");
  //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u, wsize = %u\n",levels,skip,len,skipwork,wsize);
  //   for(uint j=0; j<len; j++) printf("\nx[%u] = %g\n",j,x[j]);
  // }

  // if(threadIdx.x == 1) printf("\nhere1");
  
  // copy into shared memory
  if (i < len){
    x_work[ish] = x[i];
    x_work[ish+1] = x[i + (skip>>(levels-1))];
    // printf("\nthread.id%i - I've got i=%u less than len! Modified skip is %u. Sanity check: x[i]=%g, x[i+modskip]=%g, i+modskip=%u\n",threadIdx.x,i,skip>>(levels-1),x[i],x[i+(skip>>(levels-1))],i+(skip>>(levels-1)));
  }
  
  __syncthreads();

  // if(threadIdx.x == 1) printf("\nhere2");
  
  // if(threadIdx.x == 1){
  //   for(uint j=0; j<wsize; j++) printf("\nx_work[%u] = %g",j,x_work[j]);
  // }

  ish = threadIdx.x<<levels; //space ish for correct level of transform
  __syncthreads();

  //then we loop over a few levels

  for(li = 0; li < levels; li++){

    if (ish < wsize){
      // printf("\nthread.id%i, ish=%u, skipwork=%u, f( - ) = %g, f( + ) = %g",threadIdx.x, ish, skipwork,(x_work[ish] - x_work[ish+skipwork])*invR2,(x_work[ish] + x_work[ish+skipwork])*invR2);
      tmp = (x_work[ish] - x_work[ish+skipwork])*invR2;
      x_work[ish] = (x_work[ish] + x_work[ish+skipwork])*invR2;
      x_work[ish+skipwork] = tmp;
    }
    
    __syncthreads();
    // if(threadIdx.x == 1){
    //   printf("\nPrinting x_work....");
    //   for(uint j=0; j<wsize; j++) printf("\nx_work[%u] = %g",j,x_work[j]);
    //   printf("\nish = %u, skipwork= %u",ish,skipwork);
    // }


    skipwork=skipwork>>1; ish=ish>>1;
    __syncthreads();
  }
  
  // if(threadIdx.x == 1) printf("\nhere3");

  ish = threadIdx.x<<1;
  
  if (i < len){
    x[i] = x_work[ish];
    x[i + (skip>>(levels-1))] = x_work[ish+1];
  }

  __syncthreads();
  // if(threadIdx.x == 1) printf("\nhere4");
  
}

/*---------------------------------------------------------
  Functions for fastest GPU (CUDA) wavelet transform with added streams
   - so the same as HaarCUDAML but with streams
   - and also takes host plus device pointers, so we do memcopy
   in this function rather than outside
  ---------------------------------------------------------*/

int HaarCUDAMLv2(real* x_h, real* x_d, uint len, short int sense, uint nlevels, cudaStream_t stream){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  int ret;
  nlevels = check_len_levels(len,nlevels,filterlength);
  cudaMemcpyAsync(x_d,x_h,len*sizeof(real),HTD,stream);
  // copy x_h across to x_d
  switch(sense){
  case 1:
    ret = fHaarCUDAMLv2(x_h,x_d,len,1,nlevels,stream);
    break;
  case 0:
    ret = bHaarCUDAMLv2(x_h,x_d,len,1<<(nlevels-1),stream);
    break;
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
    break;
  }
  cudaMemcpyAsync(x_h,x_d,len*sizeof(real),DTH,stream);
  // we copy x_d back into x_h
  // we have to do this after the DWT, as the transform is in-place
  return(ret);
}

int fHaarCUDAMLv2(real* x_h, real* x_d, uint len, uint skip, uint nlevels, cudaStream_t stream){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/(skip<<1) + threadsPerBlock - 1) / threadsPerBlock;
    
    uint levels=1; //leave at 1. This is initialisation for level variable!
    
    //printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    
    while((levels+1<=2)&&((skip<<(levels+1))<=(1 << nlevels))){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }
    Haar_kernel_shared_ml_f<<<blocksPerGrid, threadsPerBlock,0,stream>>>(x_d,len,skip,levels);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fHaar ML (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    //cudaDeviceSynchronize();

    //printf("CUDA: len=%u,skip=%u\n",len,skip);
    //printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();

    return(fHaarCUDAMLv2(x_h,x_d,len,skip<<levels,nlevels,stream));

  }
  return(0);
}


int bHaarCUDAMLv2(real* x_h, real* x_d, uint len, uint skip, cudaStream_t stream){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;

    uint levels=1; //leave at 1. This is initialisation for level variable!

    //printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    
    while((levels+1<=2)&&((skip>>levels)>0)){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }
    Haar_kernel_shared_ml_b<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(x_d,len,skip,levels);

    //Haar_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bHaar ML (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    //cudaDeviceSynchronize();
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    return(bHaarCUDAMLv2(x_h, x_d,len,skip>>levels,stream));
  }
  return(0);
}



/*---------------------------------------------------------
  HaarCUDAMLv3 - like v2 but without host memory
   - so the same as HaarCUDAML but with streams
   - then we also work with just device memory
  ---------------------------------------------------------*/

int HaarCUDAMLv3(real* x_d, uint len, short int sense, uint nlevels, cudaStream_t stream){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  int ret;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case 1:
    ret = fHaarCUDAMLv3(x_d,len,1,nlevels,stream);
    break;
  case 0:
    ret = bHaarCUDAMLv3(x_d,len,1<<(nlevels-1),stream);
    break;
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
    break;
  }
  return(ret);
}

int fHaarCUDAMLv3(real* x_d, uint len, uint skip, uint nlevels, cudaStream_t stream){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/(skip<<1) + threadsPerBlock - 1) / threadsPerBlock;
    
    uint levels=1; //leave at 1. This is initialisation for level variable!
    
    //printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    
    while((levels+1<=2)&&((skip<<(levels+1))<=(1 << nlevels))){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }
    Haar_kernel_shared_ml_f<<<blocksPerGrid, threadsPerBlock,0,stream>>>(x_d,len,skip,levels);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fHaar ML (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    //cudaDeviceSynchronize();

    //printf("CUDA: len=%u,skip=%u\n",len,skip);
    //printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();

    return(fHaarCUDAMLv3(x_d,len,skip<<levels,nlevels,stream));

  }
  return(0);
}


int bHaarCUDAMLv3(real* x_d, uint len, uint skip, cudaStream_t stream){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;

    uint levels=1; //leave at 1. This is initialisation for level variable!

    //printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    
    while((levels+1<=2)&&((skip>>levels)>0)){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }
    Haar_kernel_shared_ml_b<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(x_d,len,skip,levels);

    //Haar_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bHaar ML (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    //cudaDeviceSynchronize();
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    return(bHaarCUDAMLv3(x_d,len,skip>>levels,stream));
  }
  return(0);
}



/*---------------------------------------------------------
  Functions for GPU (CUDA) wavelet transform with coalesced memory
  ---------------------------------------------------------*/

/*
  data structure used is the following:
  x is our vector

  x                        transform
  i 
  0   | 0 |   |d11|   |d11|   |d11|
  1   | 0 |   |d12|   |d12|   |d12|
  2   | 5 |   |d13|   |d13|   |d13|
  3   | 4 |   |d14|   |d14|   |d14|
  4   | 8 |   |s11|   |d21|   |d21|
  5   | 6 |   |s12|   |d22|   |d22|
  6   | 7 |   |s13|   |s21|   |d31|
  7   | 3 |   |s14|   |s22|   |s31|

  pos=0   pos=4    p=6    pos=7

  ----> forward transform ---->
  <--- backward transform <----

  where sij (dij) is the scaling (detail) coefficient number j at level i.
*/


int HaarCUDACoalA(real* x_d, uint len, short int sense){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  if(!((len != 0) && ((len & (~len + 1)) == len))){
    printf("\nLength %i is not a power of 2: ",len);
    printf("\nso we exit. Try again...\n");
    return(1);}
  switch(sense){
  case 1:
    return(fHaarCUDACA(x_d,len,0));
  case 0:
    return(bHaarCUDACA(x_d,len,len>>1));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaarCUDACA(real* x_d, uint len, uint pos){
  uint ssize = (len - pos)>>1;
  real *s;
  if(pos <(len-1)){
    // we allocate s on device
    cudaError_t cuderr = cudaMalloc((void **)&s,ssize*sizeof(real));   
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
    
    // set blocksize etc
    int threadsPerBlock = 64;
    int blocksPerGrid =(ssize + threadsPerBlock - 1) / threadsPerBlock;
    
    // run forward kernel
    Haar_CA_kernelf<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,pos,s,ssize);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fHaar (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    
    // Copy s back into x. We need seperate kernel, because we can't otherwise guarantee that 
    Coal_copy_kernelf<<<blocksPerGrid, threadsPerBlock>>>(x_d,pos,s,ssize);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fHaar (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    //free s
    cudaFree(s);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in cudaFree (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }

    cudaDeviceSynchronize();

    // printf("CUDA: len=%u,pos=%u\n",len,pos);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    return(fHaarCUDACA(x_d,len,pos+ssize));
  }
  return(0);
}

int bHaarCUDACA(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = 64;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;
    Haar_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bHaar (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    return(bHaarCUDACA(x_d,len,skip>>1));
  }
  return(0);
}

__global__ void Haar_CA_kernelf(real* x, const uint len, const uint pos, real* s, const uint ssize){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x);
  uint j = pos + 2*i;
  if (i < ssize){
    s[i] = (x[j] + x[j+1])*invR2;
    x[i+pos] = (x[j] - x[j+1])*invR2;
    // is this always going to work? j = pos + 2*i. Could have that thread 1 executes before thread 0?
    // if we store x in shared memory...could still have that thread 0, block 1 is executed before thread blockdim-1, block 0. So, still an issue. 
  }
}

__global__ void Coal_copy_kernelf(real* x, const uint pos, real* s, const uint ssize){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x);
  if(i < ssize){
    x[i+pos+ssize] = s[i];
  }
}


// ##################################################################
// Haar MODWT v1
//
// packet ordered
// using device memory throughout here
// so no CPU <-> GPU memcpy here
//
// ##################################################################


int HaarCUDAMODWT(real* x_d, real* xdat_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDAMODWT(x_d,xdat_d,len,1,nlevels));
  case BWD:
    printf("\nNot yet implemented!\n");
    return(1);
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarCUDAMODWT(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels){
  int shift, shift2, lenos, cstart, copyfrom;
  int l2s = 0; // log2(shift)
  int res;
  cudaError_t cuderr;
  //real* xdat_d; // xdat on the GPU

  // // allocate xdat on the GPU
  // cuderr = cudaMalloc((void **)&(*xdat_d),len*2*nlevels*sizeof(real));   
  // if (cuderr != cudaSuccess)
  //   {
  //     fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
  //     exit(EXIT_FAILURE);
  //   }
  // cudaDeviceSynchronize();
  
  // copy x to the start of x_dat
  // - actually, this will be done in the kernel (implicitly)
  
  // set up some CUDA variables
  int threadsPerBlock = BLOCK_SIZE;
  
  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=(skip<<1)){
    lenos=len/skip; // we pre-compute this for efficiency
    
    //loop through shifts
    for(shift=0;shift<(skip<<1);shift++){
      shift2 = shift % 2;
      
      cstart = 2*len*l2s + shift*lenos;
      copyfrom = cstart - 2*len - shift2*lenos;

      int blocksPerGrid =(lenos + threadsPerBlock - 1) / threadsPerBlock;

      // memory copy will be done in the kernel

      // do single level wvt transform on current level

      if(skip==1){
	Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock>>>(x_d,xdat_d+cstart,len,1,shift2);
      }else{
	Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock>>>(xdat_d+copyfrom,xdat_d+cstart,lenos*2,2,shift2);
      }
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in fHaarMODWT (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaDeviceSynchronize();
    }
    
    l2s++; //update log2(shift)
  }
  return(0);
}


// we should improve this to calculate 2 shifts at once in the same kernel
// seeing as they use the same data
// although it would only save us a copy from global to shared
// and reduce wasted GPU resources by a half!
__global__ void Haar_kernel_MODWT(real* x_in, real* x_out, const uint len, const uint skip, const uint shift){
  uint i_o = (blockDim.x * blockIdx.x + threadIdx.x)<<1;
  // counter inside vector x_out
  uint i_i = i_o*skip;
  // counter inside vector x_in
  uint ish = threadIdx.x<<1;
  // counter inside shared vector x_work

  __shared__ real x_work[2*BLOCK_SIZE];

  // we copy from x_in to x_out
  // but we copy indices 0, 2, ..., len - 2
  // from x_in
  // into indices 0, 1, ..., len/2 - 1
  // of x_out
  // so len is only the length of x_in
  // and the number of threads needed is len/4
  // as we need 1/2 * length(x_out) threads
  // ...
  // except for the first level of transform!
  // where skip is 1 & length(x_in) = length(x_out)
  
  if(i_i<len){
    x_work[ish] = x_in[(i_i + shift*skip) % len];
    x_work[ish+1] = x_in[(i_i + shift*skip + skip) % len];
    // load the appropiate values into shared memory

    
    // // testing our data structures
    // x_out[i_o] = x_work[ish];
    // x_out[i_o+1] = x_work[ish+1];

    x_out[i_o] = (x_work[ish] + x_work[ish+1])*invR2;
    x_out[i_o+1] = (x_work[ish] - x_work[ish+1])*invR2;
    // write the results into the output vector
  }
  // else printf("\n%u %u %u %u",i_i, len, skip, shift);

  __syncthreads();
}



// alternative version of the Haar MODWT kernel
// (packet ordered)
// that transforms both shifts of a wvt packet
__global__ void Haar_kernel_MODWT_v2(real* x_in, real* x_out, const uint len, const uint skip){
  uint i_o = (blockDim.x * blockIdx.x + threadIdx.x)<<1;
  // counter inside vector x_out
  uint i_i = i_o*skip;
  // counter inside vector x_in
  uint ish = threadIdx.x<<1;
  // counter inside shared vector x_work
  uint olen = skip==1? len: (len>>1);
  // length of the output vector

  __shared__ real x_work[2*BLOCK_SIZE+1];

  // we copy from x_in to x_out
  // but we copy indices 0, 2, ..., len - 2
  // from x_in
  // into indices 0, 1, ..., len/2 - 1
  // of x_out
  // so len is only the length of x_in
  // and the number of threads needed is len/4
  // as we need 1/2 * length(x_out) threads
  // ...
  // except for the first level of transform!
  // where skip is 1 & length(x_in) = length(x_out)
  
  if(i_i<len){
    x_work[ish] = x_in[i_i];
    x_work[ish+1] = x_in[i_i  + skip];
    if(olen < BLOCK_SIZE*2){
      // if this is the case, then we won't completely fill the shared memory
      if(i_i == len - 2*skip) x_work[ish+2] = x_in[(i_i + 2*skip)%len];
    }
    else{
      // in this case we have any extra boundary point for each shared memory block
      if(threadIdx.x == blockDim.x - 1) x_work[ish+2] = x_in[(i_i + 2*skip)%len];
    }
    // load the appropiate values into shared memory
    // we load an extra point at the end of the shared memory
    // for when shift==1 below

    
    // // testing our data structures
    // x_out[i_o] = x_work[ish];
    // x_out[i_o+1] = x_work[ish+1];
    
    x_out[i_o] = (x_work[ish] + x_work[ish+1])*invR2;
    x_out[i_o+1] = (x_work[ish] - x_work[ish+1])*invR2;
    // for shift == 0

    x_out[i_o + olen] = (x_work[ish+1] + x_work[ish+2])*invR2;
    x_out[i_o+1 + olen] = (x_work[ish+1] - x_work[ish+2])*invR2;
    // for shift == 1

    // write the results into the output vector
  }
  // else printf("\n%u %u %u %u",i_i, len, skip, shift);

  __syncthreads();
}




// ##################################################################
// Haar MODWT v2
//
// packet ordered
// using device memory throughout here
// so no CPU <-> GPU memcpy here
// also using streams!!
//
// ##################################################################


int HaarCUDAMODWTv2(real* x_d, real* xdat_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDAMODWTv2(x_d,xdat_d,len,1,nlevels));
  case BWD:
    printf("\nNot yet implemented!\n");
    return(1);
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarCUDAMODWTv2(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels){
  int shift, shift2, lenos, cstart, copyfrom;
  int l2s = 0; // log2(shift)
  int res;
  cudaError_t cuderr;
  cudaStream_t stream[2];
  for (shift = 0; shift<2; shift++)
    cudaStreamCreate(&stream[shift]);
  //create 2 streams!

  //real* xdat_d; // xdat on the GPU

  // // allocate xdat on the GPU
  // cuderr = cudaMalloc((void **)&(*xdat_d),len*2*nlevels*sizeof(real));   
  // if (cuderr != cudaSuccess)
  //   {
  //     fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
  //     exit(EXIT_FAILURE);
  //   }
  // cudaDeviceSynchronize();
  
  // copy x to the start of x_dat
  // - actually, this will be done in the kernel (implicitly)
  
  // set up some CUDA variables
  int threadsPerBlock = BLOCK_SIZE;
  
  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=(skip<<1)){
    lenos=len/skip; // we pre-compute this for efficiency
    
    //loop through shifts
    for(shift=0;shift<(skip<<1);shift++){
      shift2 = shift % 2;
      
      cstart = 2*len*l2s + shift*lenos;
      copyfrom = cstart - 2*len - shift2*lenos;

      int blocksPerGrid =(lenos + threadsPerBlock - 1) / threadsPerBlock;

      // memory copy will be done in the kernel

      // do single level wvt transform on current level

      cudaStreamSynchronize(stream[shift2]);
      
      if(skip==1){
	Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock,0,stream[shift2]>>>(x_d,xdat_d+cstart,len,1,shift2);
      }else{
	Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock,0,stream[shift2]>>>(xdat_d+copyfrom,xdat_d+cstart,lenos*2,2,shift2);
      }
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in fHaarMODWT (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      // cudaStreamSynchronize(stream[shift2]);
    }
    
    l2s++; //update log2(shift)
  }
  for (shift = 0; shift<2; shift++)
    cudaStreamDestroy(stream[shift]);
  return(0);
}



// ##################################################################
// Haar MODWT v3
//
// packet ordered
// using host memory input
// so plenty CPU <-> GPU memcpy here
// also using streams!!
// so we can have some nice async memory transfers
//
// ##################################################################


int HaarCUDAMODWTv3(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDAMODWTv3(x_h,xdat_h,x_d,xdat_d,len,1,nlevels));
  case BWD:
    printf("\nNot yet implemented!\n");
    return(1);
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarCUDAMODWTv3(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, uint skip, uint nlevels){
  int shift, shift2, lenos, cstart, copyfrom;
  int l2s = 0; // log2(shift)
  int res;
  cudaError_t cuderr;
  cudaStream_t stream[2];
  for (shift = 0; shift<2; shift++)
    cudaStreamCreate(&stream[shift]);
  //create 2 streams!

  //real *x_d, *xdat_d; // x, xdat on the GPU
  
  // // allocate xdat on the GPU
  // cuderr = cudaMalloc((void **)&xdat_d,len*2*nlevels*sizeof(real));   
  // if (cuderr != cudaSuccess)
  //   {
  //     fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
  //     exit(EXIT_FAILURE);
  //   }
  // *xdat_h=(real *)malloc(len*2*nlevels*sizeof(real));
  // cudaDeviceSynchronize();
  
  cuderr = cudaMemcpy(x_d,x_h,len*sizeof(real),HTD);
  cuderr = cudaGetLastError();
  if (cuderr != cudaSuccess)
    {
      fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
      exit(EXIT_FAILURE);
    }
  
  cudaDeviceSynchronize();
  
  
  // copy x to the start of x_dat
  // - actually, this will be done in the kernel (implicitly)
  
  // set up some CUDA variables
  int threadsPerBlock = BLOCK_SIZE;
  
  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=(skip<<1)){
    lenos=len/skip; // we pre-compute this for efficiency
    
    // we try streams in infallible setting!

    //loop through shifts
    for(shift=0;shift<(skip<<1);shift+=2){
      shift2 = shift % 2;
      
      cstart = 2*len*l2s + shift*lenos;
      copyfrom = cstart - 2*len - shift2*lenos;

      int blocksPerGrid =(lenos/2 + threadsPerBlock - 1) / threadsPerBlock;

      // memory copy will be done in the kernel

      // do single level wvt transform on current level

      //cudaStreamSynchronize(stream[shift2]);
      
      cudaDeviceSynchronize();
      
      if(skip==1){
	Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock,0,stream[0]>>>(x_d,xdat_d+cstart,len,1,0);
	Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock,0,stream[1]>>>(x_d,xdat_d+cstart+lenos,len,1,1);
	// run 2 streams concurrently
	//Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock,0,stream[0]>>>(x_d,xdat_d+cstart,len,1,shift2);
      }else{
	Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock,0,stream[0]>>>(xdat_d+copyfrom,xdat_d+cstart,lenos*2,2,0);
	Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock,0,stream[1]>>>(xdat_d+copyfrom,xdat_d+cstart+lenos,lenos*2,2,1);
	//Haar_kernel_MODWT<<<blocksPerGrid,threadsPerBlock,0,stream[0]>>>(xdat_d+copyfrom,xdat_d+cstart,lenos*2,2,shift2);
      }
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in fHaarMODWT (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaMemcpyAsync(xdat_h+cstart,xdat_d+cstart,lenos*sizeof(real),DTH,stream[0]);
      cudaMemcpyAsync(xdat_h+cstart+lenos,xdat_d+cstart+lenos,lenos*sizeof(real),DTH,stream[1]);
      //cudaMemcpy(*xdat_h+cstart,xdat_d+cstart,lenos*sizeof(real),DTH);
      // cudaStreamSynchronize(stream[shift2]);
    }
    
    l2s++; //update log2(shift)
  }
  for (shift = 0; shift<2; shift++)
    cudaStreamDestroy(stream[shift]);
  cudaFree(x_d);
  cudaFree(xdat_d);
  return(0);
}

// ##################################################################
// Haar MODWT v4
//
// packet ordered
// using host memory input
// so plenty CPU <-> GPU memcpy here
// also using streams!!
// so we can have some nice async memory transfers
//
// and also computing 2 shifts per kernel
//
// ##################################################################

int HaarCUDAMODWTv4(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDAMODWTv4(x_h,xdat_h,x_d,xdat_d,len,1,nlevels));
  case BWD:
    printf("\nNot yet implemented!\n");
    return(1);
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaarCUDAMODWTv4(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, uint skip, uint nlevels){
  int shift, shift4, lenos, cstart, copyfrom;
  int l2s = 0; // log2(shift)
  int res;
  cudaError_t cuderr;
  int sn = 2; // # streams
  cudaStream_t stream[sn];
  for (shift = 0; shift<sn; shift++)
    cudaStreamCreate(&stream[shift]);
  //create sn streams!

  // real *x_d, *xdat_d; // x, xdat on the GPU

  // // allocate x_d on the GPU
  // cuderr = cudaMalloc((void **)&x_d,len*sizeof(real));   
  // if (cuderr != cudaSuccess)
  //   {
  //     fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
  //     exit(EXIT_FAILURE);
  //   }
  
  // // allocate xdat on the GPU
  // cuderr = cudaMalloc((void **)&xdat_d,len*2*nlevels*sizeof(real));   
  // if (cuderr != cudaSuccess)
  //   {
  //     fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
  //     exit(EXIT_FAILURE);
  //   }
  //*xdat_h=(real *)malloc(len*2*nlevels*sizeof(real));  
  //cudaDeviceSynchronize();
  // cudaMallocHost(xdat_h,len*2*nlevels*sizeof(real)); // pinned host memory
  
  cuderr = cudaMemcpy(x_d,x_h,len*sizeof(real),HTD);
  cuderr = cudaGetLastError();
  if (cuderr != cudaSuccess)
    {
      fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
      exit(EXIT_FAILURE);
    }
  
  //  cudaDeviceSynchronize();
  
  
  // copy x to the start of x_dat
  // - actually, this will be done in the kernel (implicitly)
  
  // set up some CUDA variables
  int threadsPerBlock = BLOCK_SIZE;
  
  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=(skip<<1)){
    lenos=len/skip; // we pre-compute this for efficiency

    //loop through shifts
    for(shift=0;shift<(skip<<1);shift+=2){
      shift4 = shift % (2*sn);
      
      cstart = 2*len*l2s + shift*lenos;
      copyfrom = cstart - 2*len;

      int blocksPerGrid =(lenos/2 + threadsPerBlock - 1) / threadsPerBlock;

      // memory copy will be done in the kernel

      // do single level wvt transform on current level

      // cudaStreamSynchronize(stream[shift4/2]);
      
      //cudaDeviceSynchronize();
      
      if(skip==1){
	Haar_kernel_MODWT_v2<<<blocksPerGrid,threadsPerBlock,0,stream[shift4/2]>>>(x_d,xdat_d+cstart,len,1);

      }else{
	Haar_kernel_MODWT_v2<<<blocksPerGrid,threadsPerBlock,0,stream[shift4/2]>>>(xdat_d+copyfrom,xdat_d+cstart,lenos*2,2);
	
      }
      // cuderr = cudaGetLastError();
      // if (cuderr != cudaSuccess)
      // 	{
      // 	  fprintf(stderr, "CUDA error in transform in fHaarMODWT (error code %s)!\n", cudaGetErrorString(cuderr));
      // 	  exit(EXIT_FAILURE);
      // 	}
      //cudaMemcpyAsync(*xdat_h+cstart,xdat_d+cstart,2*lenos*sizeof(real),DTH,stream[shift4/2]);

      //cudaMemcpy(*xdat_h+cstart,xdat_d+cstart,lenos*sizeof(real),DTH);
    }
    for (shift = 0; shift<sn; shift++)
      cudaStreamSynchronize(stream[shift]);
    // ensure that all streams are done with work before copying completed level
    if(l2s>0) cudaMemcpyAsync(xdat_h+2*len*(l2s-1),xdat_d+2*len*(l2s-1),2*len*sizeof(real),DTH,stream[sn-1]); // copy all of the new level of detail/scaling coeffs to the host
    // cuderr = cudaGetLastError();
    // if (cuderr != cudaSuccess)
    //   {
    // 	fprintf(stderr, "CUDA error in transform in async memcpy (error code %s)!\n", cudaGetErrorString(cuderr));
    // 	exit(EXIT_FAILURE);
    //   }
    l2s++; //update log2(shift)
  }
  cudaMemcpyAsync(xdat_h+2*len*(l2s-1),xdat_d+2*len*(l2s-1),2*len*sizeof(real),DTH,stream[sn-1]);
  for (shift = 0; shift<sn; shift++)
    cudaStreamDestroy(stream[shift]);
  cudaFree(x_d);
  cudaFree(xdat_d);
  return(0);
}


// ##################################################################
// Haar MODWT v5
//
// time ordered
// using device memory throughout here
// so no CPU <-> GPU memcpy here
//
// ##################################################################

int HaarCUDAMODWTv5(real* x_d, real* xdat_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDAMODWTv5(x_d,xdat_d,len,1,nlevels));
  case BWD:
    return(bHaarCUDAMODWTv5(x_d,xdat_d,len,1<<(nlevels-1),nlevels));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarCUDAMODWTv5(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels){
  int sstart, dstart, readfrom;
  int l2s = 0; // log2(shift)
  int res;
  cudaError_t cuderr;

  // // allocate xdat on the GPU
  // cuderr = cudaMalloc((void **)&(*xdat_d),len*2*nlevels*sizeof(real));   
  // if (cuderr != cudaSuccess)
  //   {
  //     fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
  //     exit(EXIT_FAILURE);
  //   }
  // cudaDeviceSynchronize();
  
  // set up some CUDA variables
  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
  
  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=(skip<<1)){
    
    sstart = len*2*l2s;
    dstart = sstart + len;
    readfrom = sstart - 2*len;
    
    if(skip==1){
      // call Haar_kernel_MODWT_v3
      // with x_in = x_d
      Haar_kernel_MODWT_v3<<<blocksPerGrid,threadsPerBlock>>>(x_d,xdat_d+sstart,xdat_d+dstart,len,1);
    }
    else{
      // call Haar_kernel_MODWT_v3
      // with x_in = xdat_d + readfrom
      Haar_kernel_MODWT_v3<<<blocksPerGrid,threadsPerBlock>>>(xdat_d+readfrom,xdat_d+sstart,xdat_d+dstart,len,skip);
    }
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in transform in fHaarMODWT (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
    // cudaDeviceSynchronize();
    
    l2s++; //update log2(shift)
  }//skip loop
  return(0);
}


int bHaarCUDAMODWTv5(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels){
  int dstart, sstart, copyto;
  int l2s = 0; // log2(shift)
  int l; // level
  int res;
  cudaError_t cuderr;
  
  int lenos = len/skip;

  // set up some CUDA variables
  int threadsPerBlock;
  int blocksPerGrid;

  int bl; // min of blocksize & len  
  int sm; // shared memory size

  // loop over l
  for(l=nlevels-1;l>=0;l--){

    // set up some CUDA variables
    threadsPerBlock = BLOCK_SIZE;
    blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
    
    bl = min(threadsPerBlock,(int)len);
    
    sm = 2*(threadsPerBlock + (bl+lenos-1)/lenos)*sizeof(real); // required shared memory
    // this is a faster version of
    // 2*(threadsPerBlock + ceil((real)bl/(real)lenos))*sizeof(real)
      
    if(skip==1){
      sstart = 0;
      dstart = len;
      copyto = 0;
      
      // run kernel
      Haar_kernel_MODWT_v3_bsh<<<blocksPerGrid,threadsPerBlock,sm>>>(xdat_d+sstart,xdat_d+dstart,x_d+copyto,len,skip);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      copyto = sstart - 2*len;
      
      // run kernel
      Haar_kernel_MODWT_v3_bsh<<<blocksPerGrid,threadsPerBlock,sm>>>(xdat_d+sstart,xdat_d+dstart,xdat_d+copyto,len,skip);
    }
    //Haar_kernel_MODWT_v3_bsh<<<blocksPerGrid,threadsPerBlock>>>(*xdat_d+sstart,*xdat_d+dstart,*xdat_d+copyto,len,skip);
    // as yet untested!
    skip=skip>>1;
    lenos = lenos<<1;
    //cudaDeviceSynchronize();
  }
  
  return(0);
  
}



// Time-ordered Haar MODWT kernel
__global__ void Haar_kernel_MODWT_v3(real* x_in, real* s_out, real* d_out, const uint len, const uint skip){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x);
  // counter inside all global device memory vectors
  uint ish = threadIdx.x<<1;
  // counter inside shared vector x_work

  __shared__ real x_work[BLOCK_SIZE*2];

  if(i < len){
    x_work[ish] = x_in[i];
    x_work[ish+1] = x_in[(i + skip)%len];
    // this will be trickier/expensive for longer filters!
    // loads the required values into x_work
    // so that shared memory reads are coalesced
    // and ensures that the i+skip value is indeed in the
    // correct block of shared memory
    // (we need this for skip>BLOCK_SIZE, for example)

    s_out[i] = (x_work[ish] + x_work[ish+1])*invR2;
    d_out[i] = (x_work[ish] - x_work[ish+1])*invR2;
    
  }

  __syncthreads();
}

// Time-ordered Haar MODWT kernel with spacing
// This one appears to be slightly slower actually
__global__ void Haar_kernel_MODWT_v3_1(real* x_in, real* s_out, real* d_out, const uint len, const uint skip){
  uint i = ((blockDim.x * blockIdx.x + threadIdx.x)*3)%len;
  // counter inside all global device memory vectors
  // spaced out by 3 so we don't have clashes in reads
  uint ish = threadIdx.x<<1;
  // counter inside shared vector x_work

  __shared__ real x_work[BLOCK_SIZE*2];

  if(i < len){
    x_work[ish] = x_in[i];
    x_work[ish+1] = x_in[(i + skip)%len];
    // this will be trickier/expensive for longer filters!
    // loads the required values into x_work
    // so that shared memory reads are coalesced
    // and ensures that the i+skip value is indeed in the
    // correct block of shared memory
    // (we need this for skip>BLOCK_SIZE, for example)

    s_out[i] = (x_work[ish] + x_work[ish+1])*invR2;
    d_out[i] = (x_work[ish] - x_work[ish+1])*invR2;
    
  }

  __syncthreads();
}


// Time-ordered Haar MODWT kernel -- inverse transform
// no shared memory!
__global__ void Haar_kernel_MODWT_v3_b(const real* s_in, const real* d_in, real* x_out, const uint len, const uint skip){
  // skip is for the input vectors
  // we write to x_out, which is the previous
  // level's scaling coefficients
  
  // version involving no shared memory - for comparison

  uint i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < len){
    x_out[i] = (s_in[i] + d_in[i] + s_in[(i - skip + len) % len] - d_in[(i - skip + len) % len]) * hinvR2;
    // averages the reconstruction from time i & time i-skip
  }

  __syncthreads();
}


// Time-ordered Haar MODWT kernel -- inverse transform
// using tricky shared memory!
__global__ void Haar_kernel_MODWT_v3_bsh(const real* s_in, const real* d_in, real* x_out, const uint len, const uint skip){
  // skip is for the input vectors
  // we write to x_out, which is the previous
  // level's scaling coefficients
  uint lenos = len/skip;
  uint mabl, mibl; //max, min of block size & len

  // tricky version involving shared memory
  extern __shared__ real x_work[];
  // dynamic shared memory. Size specified at kernel launch.
  
  // we spread out the index variables by multiplying by 3.
  // this is to prevent bank conflicts.
  // by avoiding the case of having adjacent threads attempting
  // to access the same part of memory

  if(len<BLOCK_SIZE){
    mabl = BLOCK_SIZE;
    mibl = len;
  }
  else{
    mabl = len;
    mibl = BLOCK_SIZE;
  }
  
  uint id = blockDim.x*blockIdx.x + threadIdx.x; //unique id over blocks
  uint tid = threadIdx.x; //shorter form of threadIdx.x - will probably be optimised out by compiler anyway!
  
  uint id3 = ((1*id) % mibl) + mibl*(id/mibl); // spread out by 3
  // this also ensures that
  // id3[(b-1)*blockDim.x ... b*blockDim.x]
  // contains all of
  // id[(b-1)*blockDim.x ... b*blockDim.x]
  // but in a different order
  uint tid3 = id3 % BLOCK_SIZE; // corresp tid
  
  
  //  uint i = id*skip; //intermediary step
  //  i = i + (i/len)*(1-len);
  /* old method that had potential of overflow */
  // new method below uses intermed values bounded above by BLOCK_SIZE*len/2
  
  uint i = tid*skip + id/lenos - len*(tid/lenos);
  if(lenos > BLOCK_SIZE){
    i += ((id / BLOCK_SIZE) % (lenos/BLOCK_SIZE))*BLOCK_SIZE*skip;
  }
  // i = (1) + (2) - (3) + (4), where
  // (1) spaces by skip
  // (2) add on k each time id*skip >= k*len (i.e. for each new packet)
  // (3) subtracts k*len each time tid*skip >= len*k (keep i in range 0 - (len-1) )
  // (4) if lenos>BLOCK_SIZE, we add multiples of BLOCK_SIZE*skip to keep i in full range of 0 - (len-1) rather than 0 - (BLOCK_SIZE-1)

  // now i = 
  // 0, skip, 2*skip, ..., len-skip,
  // 1, skip+1, ..., len-skip+1,
  // ...
  // skip -1, 2*skip - 1, ..., len-1
  // i tracks the indices of the input/output vectors as we want to access them

  // uint i3 = id3*skip;
  // i3 = i3 + (i3/len)*(1-len);
  // spread out version corresp to id3
  /* old method that had potential of overflow */
  // new method below uses intermed values bounded above by BLOCK_SIZE*len/2

  uint i3 = tid3*skip + id3/lenos - len*(tid3/lenos);
  if(lenos > BLOCK_SIZE){
    i3 += ((id3 / BLOCK_SIZE) % (lenos/BLOCK_SIZE))*BLOCK_SIZE*skip;
  }

  uint ish = (tid + 1 + (tid*skip)/len)*2;
  // this will be used inside our shared vector
  // 2, 4, 6, 8, ... , 2*(block_size + ceiling(skip*bl/len) - 1)
  // missing out one number every time we need the shared memory to contain
  // another pair of wavelet coefficients for boundary conditions
  
  uint ish3 = (tid3 + 1 + (tid3*skip)/len)*2;
  // spread out version corresp to id3

  if(id<len){
    
    if((tid%lenos)==0){
      // we have to add extra boundary coeffs
      //printf("\n  ## why no work? i = %u, ibdry = %u, ish = %u, tid = %u  ## \n",i,(i + len - skip)%len,ish,tid);
      x_work[ish-2] = s_in[(i + len - skip)%len];
      x_work[ish-1] = d_in[(i + len - skip)%len];
      // we add len before taking the modulo, as mod of negative number stays neg!
    }
    x_work[ish] = s_in[i];
    x_work[ish+1] = d_in[i];
  }

  __syncthreads();

  if(id3<len){
    
    x_out[i3] = (x_work[ish3] + x_work[ish3+1] + x_work[ish3-2] - x_work[ish3-1]) * hinvR2;
    //x_out[i] = (x_work[ish] + x_work[ish+1] + x_work[ish-2] - x_work[ish-1]) * hinvR2;
    // average the reconstruction from time i & time i-skip
    // & store the answer in the output vector
    
  }

  __syncthreads();
}


// ##################################################################
// Haar MODWT v6
//
// time ordered
// with a stream - but directed from caller function
// with an async mem copy GPU <--> CPU
//
// which means the caller function must allocate the memory
// i.e. host & device memory
// and pass both pointers
//
// and pass a stream pointer
//
// ##################################################################


int HaarCUDAMODWTv6(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, short int sense, uint nlevels, cudaStream_t stream){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDAMODWTv6(x_h,xdat_h,x_d,xdat_d,len,1,nlevels,stream));
  case BWD:
    return(bHaarCUDAMODWTv6(x_h,xdat_h,x_d,xdat_d,len,1<<(nlevels-1),nlevels,stream));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarCUDAMODWTv6(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, uint skip, uint nlevels, cudaStream_t stream){
  int sstart, dstart, readfrom;
  int l2s = 0; // log2(shift)
  int res;
  cudaError_t cuderr;
  // allocations are done in the calling function (hopefully)
  
  cudaMemcpyAsync(x_d,x_h,len*sizeof(real),HTD,stream);
  /*  cuderr = cudaGetLastError();
  if (cuderr != cudaSuccess)
    {
      fprintf(stderr, "CUDA error in cudaMemcpyAsync HtD (error code %s)!\n", cudaGetErrorString(cuderr));
      exit(EXIT_FAILURE);
      } */
  
  //cudaStreamSynchronize(stream);
  
  // set up some CUDA variables
  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
  
  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=(skip<<1)){
    
    sstart = len*2*l2s;
    dstart = sstart + len;
    readfrom = sstart - 2*len;
    
    if(skip==1){
      // call Haar_kernel_MODWT_v3
      // with x_in = x_d
      Haar_kernel_MODWT_v3<<<blocksPerGrid,threadsPerBlock,0,stream>>>(x_d,xdat_d+sstart,xdat_d+dstart,len,1);
    }
    else{
      // call Haar_kernel_MODWT_v3
      // with x_in = xdat_d + readfrom
      Haar_kernel_MODWT_v3<<<blocksPerGrid,threadsPerBlock,0,stream>>>(xdat_d+readfrom,xdat_d+sstart,xdat_d+dstart,len,skip);
    }
    /*    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in transform in fHaarMODWT (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
	}    */
    //cudaStreamSynchronize(stream);
    cudaMemcpyAsync(xdat_h + sstart, xdat_d + sstart,2*len*sizeof(real),DTH,stream);
    /*    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in memcopy haarcuda (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
	} */
    l2s++; //update log2(shift)
  }//skip loop
  // cudaFree(x_d);
  // cudaFree(xdat_d);
  return(0);
}


int bHaarCUDAMODWTv6(real* x_h, real* xdat_h, real* x_d, real* xdat_d, uint len, uint skip, uint nlevels, cudaStream_t stream){
  int dstart, sstart, copyto;
  int l2s = 0; // log2(shift)
  int l; // level
  int res;
  cudaError_t cuderr;
  
  int lenos = len/skip;

  // set up some CUDA variables
  int threadsPerBlock;
  int blocksPerGrid;

  int bl; // min of blocksize & len  
  int sm; // shared memory size

  cudaMemcpyAsync(xdat_d,xdat_h,len*2*nlevels*sizeof(real),HTD,stream);

  
  // loop over l
  for(l=nlevels-1;l>=0;l--){

    // set up some CUDA variables
    threadsPerBlock = BLOCK_SIZE;
    blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
    
    bl = min(threadsPerBlock,(int)len);
    
    sm = 2*(threadsPerBlock + (bl+lenos-1)/lenos)*sizeof(real); // required shared memory
    // this is a faster version of
    // 2*(threadsPerBlock + ceil((real)bl/(real)lenos))*sizeof(real)
      
    if(skip==1){
      sstart = 0;
      dstart = len;
      copyto = 0;
      
      

      // run kernel
      Haar_kernel_MODWT_v3_bsh<<<blocksPerGrid,threadsPerBlock,sm,stream>>>(xdat_d+sstart,xdat_d+dstart,x_d+copyto,len,skip);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      copyto = sstart - 2*len;
      
      

      // run kernel
      Haar_kernel_MODWT_v3_bsh<<<blocksPerGrid,threadsPerBlock,sm,stream>>>(xdat_d+sstart,xdat_d+dstart,xdat_d+copyto,len,skip);
    }
    skip=skip>>1;
    lenos = lenos<<1;

    // copy latest reconstruction coefficients across to host memory
    if(skip > 1){
      cudaMemcpyAsync(xdat_h + copyto, xdat_d + copyto,2*len*sizeof(real),DTH,stream);
    }
    else{
      cudaMemcpyAsync(x_h, x_d, len*sizeof(real),DTH,stream);
    }
  }
  
  return(0);

}


// ##################################################################
// Haar MODWT v6d
//
// time ordered
// with a stream - but directed from caller function
// with all device memory
//
// which means the caller function must allocate the memory
// only device memory
//
// and pass a stream pointer
//
// ##################################################################


int HaarCUDAMODWTv6d(real* x_d, real* xdat_d, uint len, short int sense, uint nlevels, cudaStream_t stream){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarCUDAMODWTv6d(x_d,xdat_d,len,1,nlevels,stream));
  case BWD:
    return(bHaarCUDAMODWTv6d(x_d,xdat_d,len,1<<(nlevels-1),nlevels,stream));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarCUDAMODWTv6d(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels, cudaStream_t stream){
  int sstart, dstart, readfrom;
  int l2s = 0; // log2(shift)
  int res;
  cudaError_t cuderr;
  // allocations are done in the calling function (hopefully)
    
  // set up some CUDA variables
  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
  
  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=(skip<<1)){
    
    sstart = len*2*l2s;
    dstart = sstart + len;
    readfrom = sstart - 2*len;
    
    if(skip==1){
      // call Haar_kernel_MODWT_v3
      // with x_in = x_d
      Haar_kernel_MODWT_v3<<<blocksPerGrid,threadsPerBlock,0,stream>>>(x_d,xdat_d+sstart,xdat_d+dstart,len,1);
    }
    else{
      // call Haar_kernel_MODWT_v3
      // with x_in = xdat_d + readfrom
      Haar_kernel_MODWT_v3<<<blocksPerGrid,threadsPerBlock,0,stream>>>(xdat_d+readfrom,xdat_d+sstart,xdat_d+dstart,len,skip);
    }
    l2s++; //update log2(shift)
  }//skip loop
  return(0);
}


int bHaarCUDAMODWTv6d(real* x_d, real* xdat_d, uint len, uint skip, uint nlevels, cudaStream_t stream){
  int dstart, sstart, copyto;
  int l2s = 0; // log2(shift)
  int l; // level
  int res;
  cudaError_t cuderr;
  
  int lenos = len/skip;

  // set up some CUDA variables
  int threadsPerBlock;
  int blocksPerGrid;

  int bl; // min of blocksize & len  
  int sm; // shared memory size
  
  // loop over l
  for(l=nlevels-1;l>=0;l--){

    // set up some CUDA variables
    threadsPerBlock = BLOCK_SIZE;
    blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
    
    bl = min(threadsPerBlock,(int)len);
    
    sm = 2*(threadsPerBlock + (bl+lenos-1)/lenos)*sizeof(real); // required shared memory
    // this is a faster version of
    // 2*(threadsPerBlock + ceil((real)bl/(real)lenos))*sizeof(real)
      
    if(skip==1){
      sstart = 0;
      dstart = len;
      copyto = 0;

      // run kernel
      //Haar_kernel_MODWT_v3_bsh<<<blocksPerGrid,threadsPerBlock,sm,stream>>>(xdat_d+sstart,xdat_d+dstart,x_d+copyto,len,skip);
      Haar_kernel_MODWT_v3_b<<<blocksPerGrid,threadsPerBlock,0,stream>>>(xdat_d+sstart,xdat_d+dstart,x_d+copyto,len,skip);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      copyto = sstart - 2*len;
      
      

      // run kernel
      //Haar_kernel_MODWT_v3_bsh<<<blocksPerGrid,threadsPerBlock,sm,stream>>>(xdat_d+sstart,xdat_d+dstart,xdat_d+copyto,len,skip);
      Haar_kernel_MODWT_v3_b<<<blocksPerGrid,threadsPerBlock,0,stream>>>(xdat_d+sstart,xdat_d+dstart,xdat_d+copyto,len,skip);
    }
    skip=skip>>1;
    lenos = lenos<<1;
  }
  
  return(0);

}
