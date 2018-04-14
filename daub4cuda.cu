#include "daub4cuda.cuh"
#include "daub4coeffs.h"

#define BLOCK_SIZE 256
#define BLOCK_SIZE2 258
#define BLOCK_SECT2 256
#define BLOCK_SIZE_ML2 261  //517 //261 //69
#define BLOCK_SIZE_ML2B 263 //519 //263 //71
#define BLOCK_SECT_ML2 256 //512 //256  //64

/******************************************************
Lifted Daubechies 4 code - in CUDA!
******************************************************/

int Daub4CUDA(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fDaub4CUDA(x_d,len,1,nlevels));
  case 0:
    return(bDaub4CUDA(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fDaub4CUDA(real* x_d, uint len, uint skip, uint nlevels){
  // with periodic boundary conditions
  //  if(skip < (len >> 1)){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;
    
    // call first kernel - first lifting phase
    Daub4_kernel1<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,1);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4, kernel1 (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    //call 2nd kernel
    Daub4_kernel2<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,1);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4, kernel2 (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    //call 3rd kernel
    Daub4_kernel3<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,1);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4, kernel1 (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    //call 4th kernel
    Daub4_kernel4<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,1);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4, kernel1 (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);


    return(fDaub4CUDA(x_d,len,skip<<1,nlevels));
  }      
  return(0);
}

int bDaub4CUDA(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid =(len/skip + threadsPerBlock - 1) / threadsPerBlock;

    //call 4th kernel
    Daub4_kernel4<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,0);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4, kernel1 (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    //call 3rd kernel
    Daub4_kernel3<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,0);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4, kernel1 (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    //call 2nd kernel
    Daub4_kernel2<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,0);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4, kernel2 (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // call 1st kernel
    Daub4_kernel1<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,0);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4, kernel1 (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);


    return(bDaub4CUDA(x_d,len,skip>>1));
  }
  return(0);
}

__global__ void Daub4_kernel1(real* x, const uint len, const uint skip, const short int sense){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip*2;
  if(sense==1){ //we are doing forward transform
    if(i < len){
      x[i] = x[i] + x[i+skip]*Cl0;
      // s1[l] = x[2l] + sqrt(3)*x[2l+1]
    }
  }
  else{ //we are doing backward transform
    if(i < len){
      x[i] = x[i] - x[i+skip]*Cl0;
      // x[2l] = s1[l] - sqrt(3)*d1[l]
    }
  }
}

__global__ void Daub4_kernel2(real* x, const uint len, const uint skip, const short int sense){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip*2;
  if(sense==1){ //we are doing forward transform
    if(i == 0){ //do first cycle manually
      x[skip] = x[skip] - x[0]*Cl1 - x[len-2*skip]*Cl2;
    }else{
      if(i<len){
	x[i+skip] = x[i+skip] - x[i]*Cl1 - x[i-2*skip]*Cl2;
	// d1[l] = x[2l+1] - s1[l]*sqrt(3)/4 - s1[l-1]*(sqrt(3)-2)/4
      }
    }
  }
  else{ //backward transform
    if(i == 0){ //do first cycle manually
      x[skip] = x[skip] + x[0]*Cl1 + x[len-2*skip]*Cl2;
    }else{
      if(i<len){
	x[i+skip] = x[i+skip] + x[i]*Cl1 + x[i-2*skip]*Cl2;
	// x[2l+1] = d1[l] + s1[l]*sqrt(3)/4 + s1[l-1]*(sqrt(3)-2)/4
      }
    }
  }
}

__global__ void Daub4_kernel3(real* x, const uint len, const uint skip, const short int sense){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip*2;
  if(sense==1){ //we are doing forward transform
    if(i<len-2*skip){
      x[i] = x[i] - x[i+3*skip];
      // s2[l] = s1[l] - d1[l+1]
    }else{ //do last cycle manually
      if(i==len-2*skip){
	x[i] = x[i] - x[skip];
      }
    }
  }
  else{ //backward transform
    if(i<len-2*skip){
      x[i] = x[i] + x[i+3*skip];
      // s1[l] = s2[l] + d1[l+1]
    }else{ //do last cycle manually
      if(i==len-2*skip){
	x[i] = x[i] + x[skip];
      }
    }
  }
}

__global__ void Daub4_kernel4(real* x, const uint len, const uint skip, const short int sense){
  uint i = (blockDim.x * blockIdx.x + threadIdx.x)*skip*2;
  if(sense==1){ //we are doing forward transform
    if(i < len){
      x[i] = x[i]*Cl3;
      // s[l] = s2[l]*(sqrt(3)-1)/sqrt(2)
      x[i+skip] = x[i+skip]*Cl4;
      // d[l] = d1[l]*(sqrt(3)+1)/sqrt(2)
    }
  }else{
    if(i < len){
      x[i] = x[i]*Cl4;
      // s2[l] = s[l]*(sqrt(3)+1)/sqrt(2)
      x[i+skip] = x[i+skip]*Cl3;
      // d1[l] = d[l]*(sqrt(3)-1)/sqrt(2)  
    }
  }
}

/*---------------------------------------------------------
  Take 2 - using shared memory & doing all 4 prev kernels in 1
  ---------------------------------------------------------*/

/* At each stage we will need in shared memory, eg

   initial
   ish      x
   0        7
   1        3
   -----------
   2        0
   3        0
   4        5
   5        4
   6        8
   7        6
   8        7
   9        3
   -----------
   10       0
   11       0

   i.e. will need to keep at edges boundary condns or start/end points from points outside BLOCK_SIZE<<1 range

   BLOCK_SIZE2 = 2^power + 2
   BLOCK_SECT2 = 2^power
   Shared memory size = 2*BLOCK_SIZE2


*/

// daub4 specific get bdrs function
__global__ void get_bdrs_sh(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BS_BD * blockIdx.x + threadIdx.x)<<1;
  int i4 = (BS_BD * blockIdx.x + threadIdx.x) << 2;
  // if(len/skip>=BLOCK_SECT2){
  if(i < lenb>>1){
    if(i4==0){
      bdrs[i4] = x[len - (2*skip)];
      bdrs[i4+1] = x[len - skip];
    }
    else{
      bdrs[i4] = x[skip*(BLOCK_SECT2*i - 2)];
      bdrs[i4+1] = x[skip*(BLOCK_SECT2*i - 1)];
    }

    if(i4==lenb-4){
      bdrs[i4+2] = x[0];
      bdrs[i4+3] = x[skip];
    }
    else{
      bdrs[i4+2] = x[skip*(BLOCK_SECT2*(i+2))];
      bdrs[i4+3] = x[skip*(BLOCK_SECT2*(i+2) + 1)];
    }
    
  }

  __syncthreads();
}

int Daub4CUDA_sh(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fDaub4CUDAsh(x_d,len,1,nlevels));
  case 0:
    return(bDaub4CUDAsh(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fDaub4CUDAsh(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE2;
    int blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
    int res;

    real* bdrs; // vector of boundary points - ensures independence of loops
    uint lenb = max((len<<1)/(skip*BLOCK_SECT2),4); // length of bdrs vector
    int tPB_bd = BS_BD;
    int bPG_bd = max(((lenb>>2) + BS_BD - 1) / BS_BD,1);
    cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
    get_bdrs_sh<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb); //we copy the boundary points into a vector
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
 
    // printveccu<<<1,1>>>(bdrs,lenb);
    // cudaDeviceSynchronize();
    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    Daub4_kernel_shared_f<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4 sh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    
    cudaFree(bdrs);

    res=fDaub4CUDAsh(x_d,len,skip<<1,nlevels);
    cudaDeviceSynchronize();
    return(res);
  }
  return(0);
}

int bDaub4CUDAsh(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE2;
    int blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
    //int res;
    
    real* bdrs; // vector of boundary points - ensures independence of loops
    uint lenb = max((len<<1)/(skip*BLOCK_SECT2),4); // length of bdrs vector
    int tPB_bd = BS_BD;
    int bPG_bd = max(((lenb>>2) + BS_BD - 1) / BS_BD,1);
    cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
    get_bdrs_sh<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb); //we copy the boundary points into a vector
    // for this version of the transform, we can share the forwards get_bdrs function for the backwards trans
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    Daub4_kernel_shared_b<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bDaub4 sh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    cudaFree(bdrs);
    
    return(bDaub4CUDAsh(x_d,len,skip>>1));
  }
  return(0);
}


__global__ void Daub4_kernel_shared_f(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT2 * blockIdx.x + threadIdx.x -1)*skip<<1;
  // i = -2*skip, 0, 2*skip, ... , len-2*skip, len
  // for each block, we have, e.g. i = -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT2-2*skip, 2*BLOCK_SECT2*skip
  int ish = (threadIdx.x)<<1;
  // ish = 0, 2,... , 2*BLOCK_SECT2-2, 2*BLOCK_SECT2 2*BLOCK_SECT2+2
  __shared__ real x_work[BLOCK_SIZE2<<1];

  // printf("\nthreadIdx.x=%i,blockIdx.x=%i,i=%i,ish=%i\n",threadIdx.x,blockIdx.x,i,ish);

  //###delete this when it's working!###
  // if(threadIdx.x == 0){
  //   for(uint j=0; j<(BLOCK_SIZE2<<1); j++) x_work[j] = 900+j;
  // }

  // First, we copy x into shared array x_work
  // NB - conditioning on threadIdx.x, not ish!
  if(threadIdx.x == 0){ //ish == 0
    if(i<(int)len){
      //      printf("here2 thread%i\n",threadIdx.x);
      // x_work[ish] = x[i];
      // x_work[ish+1] = x[i + skip];
      x_work[ish] = bdrs[blockIdx.x<<2];
      x_work[ish+1] = bdrs[1+(blockIdx.x<<2)];
    }
  }
  
  if((threadIdx.x > 0) && (threadIdx.x < BLOCK_SECT2+1)){
    // needs to be conditional on i and BLOCK_SECT2
    if(i < (int)len){
      // printf("here3a thread%i\n",threadIdx.x);
      x_work[ish] = x[i];
      x_work[ish+1] = x[i + skip];
    }
    else if(i==(int)len){
      // printf("here3b thread%i\n",threadIdx.x);
      // x_work[ish] = x[0];
      // x_work[ish+1] = x[skip];
      x_work[ish] = bdrs[2+(blockIdx.x<<2)];
      x_work[ish+1] = bdrs[3+(blockIdx.x<<2)];
    }
  }
  else if(threadIdx.x == BLOCK_SECT2+1){
    if(i<=(int)len){
      x_work[ish] = bdrs[2+(blockIdx.x<<2)];
      x_work[ish+1] = bdrs[3+(blockIdx.x<<2)];
    }
  }
  
  __syncthreads();

  //if(i==len-2) printf("X_CUDA0[%i] = %g\n",len-2,x_work[ish]);

  // if(threadIdx.x == 1){
  //   //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //   for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
  // }
  
  // __syncthreads();

  //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //if(ish<=(BLOCK_SECT2<<1)){ // leaves 1 thread idle 
  //printf("herecyc1 thread%i\n",threadIdx.x);
  x_work[ish] = x_work[ish] + x_work[ish + 1]*Cl0;
  //    x[i] = x[i] + x[i+skip]*Cl0;
  // s1[l] = x[2l] + sqrt(3)*x[2l+1]
  //}
  __syncthreads();

  // if(i>len - 9){
  //   printf("c1 X[%i]=%g\n",i,x_work[ish]);
  // }
  // __syncthreads();

  //   if(threadIdx.x == 1){
  //     //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //     for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
  //   }

  // __syncthreads();

  // if(i==len-2) printf("X_CUDA1[%i] = %g\n",len-2,x_work[ish]);

  //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(ish>=2){ // leaves 1 thread idle
    // printf("herecyc2 thread%i\n",threadIdx.x);
    x_work[ish+1] = x_work[ish+1] - x_work[ish]*Cl1 - x_work[ish-2]*Cl2;
    //    x[i+skip] = x[i+skip] - x[i]*Cl1 - x[i-2*skip]*Cl2;
    // d1[l] = x[2l+1] - s1[l]*sqrt(3)/4 - s1[l-1]*(sqrt(3)-2)/4
  }
  __syncthreads();

  // if(i>(int) len - 9){
  //   printf("c2 X[%i]=%g\n",i,x_work[ish]);
  // }
  // __syncthreads();  

  //   if(threadIdx.x == 1){
  //     //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //     for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
  //   }
  // __syncthreads();

  //if(i==len-2) printf("X_CUDA2[%i] = %g\n",len-2,x_work[ish]);
  
  //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
    //  printf("herecyc3 thread%i\n",threadIdx.x);
    x_work[ish] = x_work[ish] - x_work[ish+3];
    //    x[i] = x[i] - x[i+3*skip];
    // s2[l] = s1[l] - d1[l+1]
  }
  __syncthreads();

  // if(i>len - 9){
  //   printf("c3 X[%i]=%g\n",i,x_work[ish]);
  // }
  // __syncthreads();

  //   if(threadIdx.x == 1){
  //     //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //     for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
  //   }
  // __syncthreads();

  //  if(i==len-2) printf("X_CUDA3[%i] = %g\n",len-2,x_work[ish]);

  //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
    //  printf("herecyc4 thread%i\n",threadIdx.x);
    x_work[ish] = x_work[ish]*Cl3;
    //    x[i] = x[i]*Cl3;
    // s[l] = s2[l]*(sqrt(3)-1)/sqrt(2)
    x_work[ish+1] = x_work[ish+1]*Cl4;
    //    x[i+skip] = x[i+skip]*Cl4;
    // d[l] = d1[l]*(sqrt(3)+1)/sqrt(2)
  }
  __syncthreads();

  // if(i==len-2) printf("X_CUDA4[%i] = %g\n",len-2,x_work[ish]);

  // if(i>len - 9){
  //   printf("c4 X[%i]=%g\n",i,x_work[ish]);
  // }
  // __syncthreads();

  //   if(threadIdx.x == 1){
  //     //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //     for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);

  // }
  // __syncthreads();

  // Now transform level is done. We copy shared array x_work back into x
  //  if((i>=0)&&(i<len)){
  if((ish>=2) && (ish<(BLOCK_SIZE2<<1)-2) && (i<(int)len)){ //i>=0 as ish>=2
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skip] = x_work[ish+1];
  }

  __syncthreads();
  
}

__global__ void Daub4_kernel_shared_b(real* x, const uint len, const uint skip, real*bdrs, const uint lenb){
  int i = (BLOCK_SECT2 * blockIdx.x + threadIdx.x -1)*skip<<1;
  // i = -2*skip, 0, 2*skip, ... , len-2*skip, len
  // for each block, we have, e.g. i = -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT2-2*skip, 2*BLOCK_SECT2*skip
  int ish = (threadIdx.x)<<1;
  // ish = 0, 2,... , 2*BLOCK_SECT2-2, 2*BLOCK_SECT2 2*BLOCK_SECT2+2
  __shared__ real x_work[BLOCK_SIZE2<<1];

  //###delete this when it's working!###
  // if(threadIdx.x == 0){
  //   for(uint j=0; j<(BLOCK_SIZE2<<1); j++) x_work[j] = 900+j;
  // }

  // First, we copy x into shared array x_work
  // NB - conditioning on threadIdx.x, not ish!
  if(threadIdx.x == 0){ //ish == 0
    if(i<(int)len){
      //      printf("here2 thread%i\n",threadIdx.x);
      // x_work[ish] = x[i];
      // x_work[ish+1] = x[i + skip];
      x_work[ish] = bdrs[blockIdx.x<<2];
      x_work[ish+1] = bdrs[1+(blockIdx.x<<2)];
    }
  }
  
  if((threadIdx.x > 0) && (threadIdx.x < BLOCK_SECT2+1)){
    // needs to be conditional on i and BLOCK_SECT2
    if(i < (int)len){
      // printf("here3a thread%i\n",threadIdx.x);
      x_work[ish] = x[i];
      x_work[ish+1] = x[i + skip];
    }
    else if(i==(int)len){
      // printf("here3b thread%i\n",threadIdx.x);
      // x_work[ish] = x[0];
      // x_work[ish+1] = x[skip];
      x_work[ish] = bdrs[2+(blockIdx.x<<2)];
      x_work[ish+1] = bdrs[3+(blockIdx.x<<2)];
    }
  }
  else if(threadIdx.x == BLOCK_SECT2+1){
    if(i<=(int)len){
      //	printf("here5 thread%i\n",threadIdx.x);
      // x_work[ish] = x[i];
      // x_work[ish+1] = x[i + skip];
	
      x_work[ish] = bdrs[2+(blockIdx.x<<2)];
      x_work[ish+1] = bdrs[3+(blockIdx.x<<2)];

    }
  }

  __syncthreads();

  //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
  //  printf("herecyc4 thread%i\n",threadIdx.x);
  x_work[ish] = x_work[ish]*Cl4;
  //    x[i] = x[i]*Cl4;
  // s[l] = s2[l]*(sqrt(3)+1)/sqrt(2)
  x_work[ish+1] = x_work[ish+1]*Cl3;
  //    x[i+skip] = x[i+skip]*Cl3;
  // d[l] = d1[l]*(sqrt(3)-1)/sqrt(2)
  //  }
  __syncthreads();
  
  //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
  if(ish<=(BLOCK_SECT2<<1)){
    //  printf("herecyc3 thread%i\n",threadIdx.x);
    x_work[ish] = x_work[ish] + x_work[ish+3];
    //    x[i] = x[i] + x[i+3*skip];
    // s2[l] = s1[l] + d1[l+1]
  }
  __syncthreads();

  //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(ish>=2){ // leaves 1 thread idle
    // printf("herecyc2 thread%i\n",threadIdx.x);
    x_work[ish+1] = x_work[ish+1] + x_work[ish]*Cl1 + x_work[ish-2]*Cl2;
    //    x[i+skip] = x[i+skip] + x[i]*Cl1 + x[i-2*skip]*Cl2;
    // d1[l] = x[2l+1] + s1[l]*sqrt(3)/4 + s1[l-1]*(sqrt(3)-2)/4
  }
  __syncthreads();

  //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle 
    //printf("herecyc1 thread%i\n",threadIdx.x);
    x_work[ish] = x_work[ish] - x_work[ish + 1]*Cl0;
    //    x[i] = x[i] - x[i+skip]*Cl0;
    // s1[l] = x[2l] - sqrt(3)*x[2l+1]
  }
  __syncthreads();
  
  // Now transform level is done. We copy shared array x_work back into x
  //  if((i>=0)&&(i<len)){
  if((ish>=2) && (ish<(BLOCK_SIZE2<<1)-2) && (i<len)){ //i>=0 as ish>=2
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skip] = x_work[ish+1];
  }

  __syncthreads();
}


// ########################################################################
// Now we do the same again but with a kernel that performs multiple layers
// of transform - like the Haar kernel
// ########################################################################

/*
  Shared memory has following, new, structure:

   _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ 
  |        |                                            |            |
  |_ _ _ _ |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ |_ _ _ _ _ _ |

  |<--4--->|<---------(2*BLOCK_SECT_ML2)--------------->|<---6------>|
  |<---------------(  2 * BLOCK_SIZE_ML2  )------------------------->|

  indices...
  |    2 3 |4 5 6 ....                                k |k1k2k3k4k5k6|
  used in 1st level of transform (where k=2*BLOCK_SECT_ML2+3) k1 is k+1 etc

  then indices...
  |0 1 2 3 |4 5 6 ....                                k |k1k2k3k4k5k6|
  used in 2nd level of transform

  First & last 4/6 coefficients contain shared memory boundary coefficients* for the transform levels.
  *shared memory boundary coefficients: for the first & last shared memory blocks, they hold periodic boundary coefficient points; for all other blocks, they hold the boundary coefficients of the previous/following memory block.

  The threads point to (via the variable ish):
  (ish is actual index in shared memory)
  (skipwork = 1)
  |0   1   |2   3 ...                               l   |l1  l2  l3  |
  where l is BLOCK_SECT_ML2+1
  (skipwork = 2)
  |0       |1     ...                           m       |m1      m2  |
  where m is floor[(BLOCK_SECT_ML2+1)/2]
*/


int Daub4CUDA_sh_ml2(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fDaub4CUDAsh_ml2(x_d,len,1,nlevels));
  case 0:
    return(bDaub4CUDAsh_ml2(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fDaub4CUDAsh_ml2(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock;
    int blocksPerGrid;
 
    uint levels=1; //leave at 1. This is initialisation for level variable!

    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    while((levels+1<=2)&&((skip<<(levels+1))<=(1 << nlevels))){ // levels+1<=k gives L, #levels to loop over
      // take skip to power levels+1 as filter of length 4
      levels+=1;
    }
    

    if (levels==1){
      // deal with bdrs
      real* bdrs; // vector of boundary points - ensures independence of loops
      uint lenb = max((len<<1)/(skip*BLOCK_SECT2),4); // length of bdrs vector
      int tPB_bd = BS_BD;
      int bPG_bd = max(((lenb>>2) + BS_BD - 1) / BS_BD,1);
      cudaMalloc((void **)&bdrs,lenb*sizeof(real));


      get_bdrs_sh<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb); //we copy the boundary points into a vector
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaDeviceSynchronize();
      
      threadsPerBlock = BLOCK_SIZE2;
      blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;      

      Daub4_kernel_shared_f<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

      cudaFree(bdrs);

    }
    else{
      // deal with bdrs
      real* bdrs; // vector of boundary points - ensures independence of loops
      uint k = 6;
      uint lenb = max((len*k)/(skip*BLOCK_SECT_ML2),2*k); // length of bdrs vector
      int tPB_bd = BS_BD;
      int bPG_bd = max(((lenb/(2*k)) + BS_BD - 1) / BS_BD,1);
      cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
      get_bdrs_sh_k<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb,k,BLOCK_SECT_ML2); //we copy the boundary points into a vector
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaDeviceSynchronize();
      
      threadsPerBlock = BLOCK_SIZE_ML2;
      blocksPerGrid =(len/(skip<<1) + BLOCK_SECT_ML2 - 1) / BLOCK_SECT_ML2;

      // printf("\nlevels=2");
      // printveccu<<<1,1>>>(bdrs,lenb);
      // cudaDeviceSynchronize();
      // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);
                  
      Daub4_kernel_shared_f_ml2<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);
      cudaFree(bdrs);

    }

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4 sh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);

    cudaDeviceSynchronize();
    
    return(fDaub4CUDAsh_ml2(x_d,len,skip<<levels,nlevels));

  }
  return(0);
}

int bDaub4CUDAsh_ml2(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock;
    int blocksPerGrid;
    real* bdrs;
    
    uint levels=1; //leave at 1. This is initialisation for level variable!

    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    while((levels+1<=2)&&((skip>>levels)>0)){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }
    
    if (levels==1){
      threadsPerBlock = BLOCK_SIZE2;
      blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
      uint lenb = max((len<<1)/(skip*BLOCK_SECT2),4); // length of bdrs vector
      int tPB_bd = BS_BD;
      int bPG_bd = max(((lenb>>2) + BS_BD - 1) / BS_BD,1);
      cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
      get_bdrs_sh<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb); //we copy the boundary points into a vector
      // for this version of the transform, we can share the forwards get_bdrs function for the backwards trans
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaDeviceSynchronize();
    
      // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

      Daub4_kernel_shared_b<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

      cudaFree(bdrs);
    }
    else{
      threadsPerBlock = BLOCK_SIZE_ML2B;
      blocksPerGrid =(len/skip + BLOCK_SECT_ML2 - 1) / BLOCK_SECT_ML2;
      uint k = 8;
      uint lenb = max((len*k*2)/(skip*BLOCK_SECT_ML2),2*k); // length of bdrs vector
      //this is double the equivalent line for forward as we need it for skip/2
      
      int tPB_bd = BS_BD;
      int bPG_bd = max(((lenb/(2*k)) + BS_BD - 1) / BS_BD,1);
      cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
      get_bdrs_sh_k<<<bPG_bd, tPB_bd>>>(x_d,len,skip>>1,bdrs,lenb,k,BLOCK_SECT_ML2);
      
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaDeviceSynchronize();

      // printf("\nlevels=2");
      // printveccu<<<1,1>>>(bdrs,lenb);
      // cudaDeviceSynchronize();
      // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

      Daub4_kernel_shared_b_ml2<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);
      cudaFree(bdrs);
    }
    
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in transform in bDaub4 sh (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    
    // //print stuff...
    // printf("\nCUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    
    //cudaDeviceSynchronize();

    return(bDaub4CUDAsh_ml2(x_d,len,skip>>levels));
    
  }
  return(0);
}


__global__ void Daub4_kernel_shared_f_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT_ML2 * blockIdx.x + threadIdx.x -2)*skip << 1;
  // i = -4*skip, -2*skip, 0, 2*skip, ... , len-2*skip, len, len +2
  // for each block, we have, e.g. i = -4*skip, -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT_ML2-2*skip, 2*BLOCK_SECT_ML2*skip, 2*skip*BLOCK_SECT_ML2+2*skip
  uint ish = (threadIdx.x)<<1;
  uint ishlast = (BLOCK_SIZE_ML2-1)<<1;
  uint li;
  // ish = 0, 2,... , 2*BLOCK_SECT_ML2-2, 2*BLOCK_SECT_ML2, 2*BLOCK_SECT_ML2+2, 2*BLOCK_SECT_ML2+4, 2*BLOCK_SECT_ML2+6, +8
  __shared__ real x_work[BLOCK_SIZE_ML2<<1];
  uint skipwork = 1;
  
  // printf("\nthreadIdx.x=%i,blockIdx.x=%i,i=%i,ish=%i\n",threadIdx.x,blockIdx.x,i,ish);

  // //###delete this when it's working!###
  // if(threadIdx.x == 0){
  //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) x_work[j] = 99900+j;
  // }

  // First, we copy x into shared array x_work
  // NB - conditioning on threadIdx.x, not ish!

  //## we copy values in for the first level of the transform

  //## bdrs contains 6 values at the start & end of each block => 12 in total
  //## We only use 4 at the start & 6 at the end.

  if(threadIdx.x <= 1){ //ish == 0 or 2
    if(i<(int) len){ //periodic boundary conditions
      // printf("here1 thread%i\n",threadIdx.x);
      x_work[ish] = bdrs[ish+2+blockIdx.x*12];
      x_work[ish+1] = bdrs[ish+3+blockIdx.x*12];
      // we don't use the first 2 elements of each block of 12 bdrs values
    }
  }
  
 //## this is the main section of the shared block
  if((threadIdx.x > 1) && (threadIdx.x < BLOCK_SECT_ML2+2)){
    // needs to be conditional on i and BLOCK_SECT
    if(i < len){
      // printf("here3a thread%i\n",threadIdx.x);
      x_work[ish] = x[i];
      x_work[ish+1] = x[i + skip];
    }
    else if((i>=len)&&(i<=len+4*skip)){
      // ## then we are filling out boundary points in shared where length shared vector > len
      // printf("here3b thread%i\n",threadIdx.x);
      x_work[ish] = bdrs[(i-len)/skip + 6 + blockIdx.x*12];
      x_work[ish+1] = bdrs[(i-len)/skip + 7 + blockIdx.x*12];
      // we pick up last 6 bdrs from each block
      // i.e. this should pick up blockIdx.x*12 + {6,7,8,9,10,11}
    }
  }
  //## boundary coeffs
  else if(threadIdx.x >= BLOCK_SECT_ML2+2){
    if((i>=len)&&(i<=len+4*skip)){
      //  printf("here4 thread%i\n",threadIdx.x);
      x_work[ish] = bdrs[ish - (BLOCK_SECT_ML2<<1) - 4 + 6 + blockIdx.x*12];
      x_work[ish+1] = bdrs[ish - (BLOCK_SECT_ML2<<1) - 4 + 7 + blockIdx.x*12];
      //we pick up last 6 bdrs from each block
      //here, ish - (BLOCK_SECT_ML2<<1)  - 4 = (i-len)/skip
      //but the former should be faster to calculate!
    }
    else{
      if(i<len){
	//	printf("here5 thread%i\n",threadIdx.x);
	x_work[ish] = bdrs[(i/skip - (BLOCK_SECT_ML2<<1)*(blockIdx.x+1)) + 6 + blockIdx.x*12];
	x_work[ish+1] = bdrs[(i/skip - (BLOCK_SECT_ML2<<1)*(blockIdx.x+1)) +7+blockIdx.x*12];
      }
    }
  }  
  __syncthreads();

  // if(threadIdx.x == 1){
  //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
  //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
  // }
  // __syncthreads();
  

  //we loop over a few levels...
  for(li = 0; li < 2; li++){
    
    //    if (ish < (BLOCK_SIZE2<<1)){ ## something like this!
    //## here we must restrict level 1 but not level 2

    //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast))||
	((li==1)&&(ish<=ishlast)) ){
      //printf("herecyc1 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish] + x_work[ish + skipwork]*Cl0;
      //    x[i] = x[i] + x[i+skip]*Cl0;
      // s1[l] = x[2l] + sqrt(3)*x[2l+1]
    }
    __syncthreads();

    // if(threadIdx.x == 1){
    //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
    //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
    // }
    // __syncthreads();

    
    //## different ifs for different levels.
    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    //   if( ((li==0)&&(ish>=4)&&(ish<(BLOCK_SIZE_ML2<<1)-2))||
    if( ((li==0)&&(ish>=2)&&(ish<=ishlast))||
	((li==1)&&(ish>=4)&&(ish<ishlast)) ){
      //    if(ish>=2){ // leaves 1 thread idle
      // printf("herecyc2 thread%i\n",threadIdx.x);
      x_work[ish+skipwork] = x_work[ish+skipwork] - x_work[ish]*Cl1 - x_work[ish-2*skipwork]*Cl2;
      //    x[i+skip] = x[i+skip] - x[i]*Cl1 - x[i-2*skip]*Cl2;
      // d1[l] = x[2l+1] - s1[l]*sqrt(3)/4 - s1[l-1]*(sqrt(3)-2)/4
    }
    __syncthreads();

    // if(threadIdx.x == 1){
    //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
    //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
    // }
    // __syncthreads();

  
    //## different ifs for different levels.
    //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    //if( ((li==0)&&(ish>=4)&&(ish<=(BLOCK_SIZE_ML2<<1)-4))||
    if( ((li==0)&&(ish<ishlast))||
	((li==1)&&(ish>=4)&&(ish<=ishlast-6)) ){
      //if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
      //  printf("herecyc3 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish] - x_work[ish+3*skipwork];
      //    x[i] = x[i] - x[i+3*skip];
      // s2[l] = s1[l] - d1[l+1]
    }
    __syncthreads();

    // if(threadIdx.x == 1){
    //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
    //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
    // }
    // __syncthreads();


    //## different ifs for different levels.
    //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    //    if( ((li==0)&&(ish>=4)&&(ish<=(BLOCK_SIZE_ML2<<1)-4))||
    if( ((li==0)&&(ish<=ishlast))||
	((li==1)&&(ish>=4)&&(ish<=ishlast-6)) ){
      // if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
      //  printf("herecyc4 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish]*Cl3;
      //    x[i] = x[i]*Cl3;
      // s[l] = s2[l]*(sqrt(3)-1)/sqrt(2)
      x_work[ish+skipwork] = x_work[ish+skipwork]*Cl4;
      //    x[i+skip] = x[i+skip]*Cl4;
      // d[l] = d1[l]*(sqrt(3)+1)/sqrt(2)
    }
    __syncthreads();

    // if(threadIdx.x == 1){
    //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
    //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
    // }
    // __syncthreads();


    ish=ish<<1; skipwork=skipwork<<1;
    //__syncthreads();
    
  }
  

  // Now transform level is done. We copy shared array x_work back into x

  ish = (threadIdx.x)<<1;

  //  if((ish>=2) && (ish<(BLOCK_SIZE2<<1)-2) && (i<len)){ //i>=0 as ish>=2
  if( (ish>=4) && (ish<((BLOCK_SIZE_ML2<<1)-6)) && (i<len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skip] = x_work[ish+1];
  }

}

/*
  For backward transform, shared memory has slightly different (longer) structure:

   _ _ _ _ _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ 
  |                |                                            |            |
  |_ _ _ _ _ _ _ _ |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ |_ _ _ _ _ _ |

  |<--8----------->|<---------(2*BLOCK_SECT_ML2)--------------->|<---6------>|
  |<-----------------------(  2 * BLOCK_SIZE_ML2B )------------------------->|

*/

__global__ void Daub4_kernel_shared_b_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT_ML2 * blockIdx.x + threadIdx.x -4)*skip;
  // i = -2*skip, -skip,  0, ..., len-skip, len, len+1
  uint ish = (threadIdx.x)<<1;
  uint ishlast = min(len/(skip>>1)+10,(BLOCK_SIZE_ML2B-1)<<1); //size of shared vec x_work
  uint li;
  __shared__ real x_work[BLOCK_SIZE_ML2B<<1];
  uint skipwork = 2;

  uint skipbl = skip >>1; //actually, we need skip to be the skip of the 2nd layer of the transform, as that is the detail needed in xwork

  // //###delete this when it's working!###
  // if(threadIdx.x == 0){
  //   for(uint j=0; j<(BLOCK_SIZE_ML2B<<1); j++) x_work[j] = 99900+j;
  // }
  __syncthreads();

  //## we copy values in for the first level of the transform

  if(threadIdx.x <= 3){ //ish == 0, 2, 4 or 6
    if(i<(int) len){ //periodic boundary conditions
      // printf("here1 thread%i\n",threadIdx.x);
      x_work[ish] = bdrs[ish+blockIdx.x*16];
      x_work[ish+1] = bdrs[ish+1+blockIdx.x*16];
    }
  }
  
  //## this is the main section of the shared block
  if((threadIdx.x > 3) && (threadIdx.x < BLOCK_SECT_ML2+4)){
    // needs to be conditional on i and BLOCK_SECT
    if(i < len){
      // printf("here3a thread%i\n",threadIdx.x);
      x_work[ish] = x[i];
      x_work[ish+1] = x[i + skipbl];
    }
    else if((i>=len)&&(i<=len+4*skipbl)){
      // ## then we are filling out boundary points in shared where length shared vector > len
      // printf("here3b thread%i\n",threadIdx.x);
      x_work[ish] = bdrs[(i-len)/skipbl + 8 + blockIdx.x*16];
      x_work[ish+1] = bdrs[(i-len)/skipbl + 9 + blockIdx.x*16];
      // we pick up last 6 bdrs from each block
      // i.e. this should pick up blockIdx.x*12 + {6,7,8,9,10,11}
    }
  }
  //## boundary coeffs
  else if(threadIdx.x >= BLOCK_SECT_ML2+4){
    if((i>=len)&&(i<=len+4*skipbl)){
      //  printf("here4 thread%i\n",threadIdx.x);
      x_work[ish] = bdrs[ish - (BLOCK_SECT_ML2<<1) - 8 + 8 + blockIdx.x*16];
      x_work[ish+1] = bdrs[ish - (BLOCK_SECT_ML2<<1) - 8 + 9 + blockIdx.x*16];
      //we pick up last 6 bdrs from each block
      //here, ish - (BLOCK_SECT_ML2<<1)  - 4 = (i-len)/skipbl
      //but the former should be faster to calculate!
    }
    else{
      if(i<len){
	//	printf("here5 thread%i\n",threadIdx.x);
	x_work[ish] = bdrs[(i/skipbl - (BLOCK_SECT_ML2<<1)*(blockIdx.x+1)) + 8 + blockIdx.x*16];
	x_work[ish+1] = bdrs[(i/skipbl - (BLOCK_SECT_ML2<<1)*(blockIdx.x+1)) +9+blockIdx.x*16];
      }
    }
  }  
  __syncthreads();
  
  // if(threadIdx.x == 1){
  //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
  //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
  // }
  // __syncthreads();

  ish=ish<<1;
  // __syncthreads();

  //we loop over a few levels...
  for(li = 0; li < 2; li++){
    
    //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    if( ((li==0)&&(ish<ishlast))||
	((li==1)&&(ish<ishlast)) ){
      //  printf("herecyc4 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish]*Cl4;
      //    x[i] = x[i]*Cl4;
      // s[l] = s2[l]*(sqrt(3)+1)/sqrt(2)
      x_work[ish+skipwork] = x_work[ish+skipwork]*Cl3;
      //    x[i+skip] = x[i+skip]*Cl3;
      // d[l] = d1[l]*(sqrt(3)-1)/sqrt(2)
    }
    __syncthreads();
    
    //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    if( ((li==0)&&(ish<ishlast))||
	((li==1)&&(ish<ishlast)) ){
      //  printf("herecyc3 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish] + x_work[ish+3*skipwork];
      //    x[i] = x[i] + x[i+3*skip];
      // s2[l] = s1[l] + d1[l+1]
    }
    __syncthreads();
    
    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    if( ((li==0)&&(ish>=2)&&(ish<ishlast))||
	((li==1)&&(ish>=2)&&(ish<ishlast)) ){
      // printf("herecyc2 thread%i\n",threadIdx.x);
      x_work[ish+skipwork] = x_work[ish+skipwork] + x_work[ish]*Cl1 + x_work[ish-2*skipwork]*Cl2;
      //    x[i+skip] = x[i+skip] + x[i]*Cl1 + x[i-2*skip]*Cl2;
      // d1[l] = x[2l+1] + s1[l]*sqrt(3)/4 + s1[l-1]*(sqrt(3)-2)/4
    }
    __syncthreads();
  
    //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish>=2)&&(ish<ishlast))||
	((li==1)&&(ish>=2)&&(ish<ishlast)) ){
      //printf("herecyc1 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish] - x_work[ish + skipwork]*Cl0;
      //    x[i] = x[i] - x[i+skip]*Cl0;
      // s1[l] = x[2l] - sqrt(3)*x[2l+1]
    }
    __syncthreads();

    ish=ish>>1; skipwork=skipwork>>1;
    //__syncthreads();
   
  } 
  
  // Now transform level is done. We copy shared array x_work back into x
  
  ish = (threadIdx.x)<<1;
  
  if( (ish>=8) && (ish<((BLOCK_SIZE_ML2B<<1)-6)) && (i<len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skipbl] = x_work[ish+1];
  }
  
  // __syncthreads();

}


// ########################################################################
// Now we repeat the previous functions of using shared memory & 2 levels
// of memory within the one kernel, but this time we instead use a different
// input & output vector to avoid having to spend time filling a boundaries
// vector.
// ########################################################################


/* At each stage we will need in shared memory, eg

   initial
   ish      x
   0        7
   1        3
   -----------
   2        0
   3        0
   4        5
   5        4
   6        8
   7        6
   8        7
   9        3
   -----------
   10       0
   11       0

   i.e. will need to keep at edges boundary condns or start/end points from points outside BLOCK_SIZE<<1 range

   BLOCK_SIZE2 = 2^power + 2
   BLOCK_SECT2 = 2^power
   Shared memory size = 2*BLOCK_SIZE2


*/

int Daub4CUDA_sh_io(real* x_d_in, real* x_d_out, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fDaub4CUDAsh_io(x_d_in,x_d_out,len,1,nlevels));
  case 0:
    return(bDaub4CUDAsh_io(x_d_in,x_d_out,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fDaub4CUDAsh_io(real* x_d_in, real* x_d_out, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE2;
    int blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
    int res;
    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    Daub4_kernel_shared_f_io<<<blocksPerGrid, threadsPerBlock>>>(x_d_in, x_d_out,len,skip);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4 sh io (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    

    res=fDaub4CUDAsh_io(x_d_out, x_d_in,len,skip<<1,nlevels);
    cudaDeviceSynchronize();
    return(res);
  }
  return(0);
}

int bDaub4CUDAsh_io(real* x_d_in, real* x_d_out, uint len, uint skip){
  int res;
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE2;
    int blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
    //int res;
    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    Daub4_kernel_shared_b_io<<<blocksPerGrid, threadsPerBlock>>>(x_d_in, x_d_out,len,skip);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bDaub4 sh io (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    //cudaDeviceSynchronize();
    res=bDaub4CUDAsh_io(x_d_out, x_d_in,len,skip>>1);
    cudaDeviceSynchronize();
    return(res);
  }
  return(0);
}


__global__ void Daub4_kernel_shared_f_io(real* x_in, real* x_out, const uint len, const uint skip){
  int i = (BLOCK_SECT2 * blockIdx.x + threadIdx.x -1)*skip<<1;
  // i = -2*skip, 0, 2*skip, ... , len-2*skip, len
  // for each block, we have, e.g. i = -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT2-2*skip, 2*BLOCK_SECT2*skip
  int ish = (threadIdx.x)<<1;
  // ish = 0, 2,... , 2*BLOCK_SECT2-2, 2*BLOCK_SECT2 2*BLOCK_SECT2+2
  __shared__ real x_work[BLOCK_SIZE2<<1];

  // printf("\nthreadIdx.x=%i,blockIdx.x=%i,i=%i,ish=%i\n",threadIdx.x,blockIdx.x,i,ish);

  //###delete this when it's working!###
  // if(threadIdx.x == 0){
  //   for(uint j=0; j<(BLOCK_SIZE2<<1); j++) x_work[j] = 900+j;
  // }

  // First, we copy x into shared array x_work
  // NB - conditioning on threadIdx.x, not ish!
  if(threadIdx.x == 0){ //ish == 0
    if(i<0){
      x_work[ish] = x_in[len - (skip<<1)];
      x_work[ish+1] = x_in[len - skip];
    }
    else if(i<(int)len){
      //      printf("here2 thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skip];
    }
  }
  
  if((threadIdx.x > 0) && (threadIdx.x < BLOCK_SECT2+1)){
    // needs to be conditional on i and BLOCK_SECT2
    if(i < (int)len){
      // printf("here3a thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skip];
    }
    else if(i==(int)len){
      // printf("here3b thread%i\n",threadIdx.x);
      x_work[ish] = x_in[0];
      x_work[ish+1] = x_in[skip];
    }
  }
  else if(threadIdx.x == BLOCK_SECT2+1){
    if(i<(int)len){
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skip];
    }
    else if(i==(int)len){
      x_work[ish] = x_in[0];
      x_work[ish+1] = x_in[skip];
    }
  }
  
  __syncthreads();

  //if(i==len-2) printf("X_CUDA0[%i] = %g\n",len-2,x_work[ish]);

  // if(threadIdx.x == 1){
  //   //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //   for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
  // }
  
  // __syncthreads();

  //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //if(ish<=(BLOCK_SECT2<<1)){ // leaves 1 thread idle 
  //printf("herecyc1 thread%i\n",threadIdx.x);
  x_work[ish] = x_work[ish] + x_work[ish + 1]*Cl0;
  //    x[i] = x[i] + x[i+skip]*Cl0;
  // s1[l] = x[2l] + sqrt(3)*x[2l+1]
  //}
  __syncthreads();

  // if(i>len - 9){
  //   printf("c1 X[%i]=%g\n",i,x_work[ish]);
  // }
  // __syncthreads();

  //   if(threadIdx.x == 1){
  //     //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //     for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
  //   }

  // __syncthreads();

  // if(i==len-2) printf("X_CUDA1[%i] = %g\n",len-2,x_work[ish]);

  //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(ish>=2){ // leaves 1 thread idle
    // printf("herecyc2 thread%i\n",threadIdx.x);
    x_work[ish+1] = x_work[ish+1] - x_work[ish]*Cl1 - x_work[ish-2]*Cl2;
    //    x[i+skip] = x[i+skip] - x[i]*Cl1 - x[i-2*skip]*Cl2;
    // d1[l] = x[2l+1] - s1[l]*sqrt(3)/4 - s1[l-1]*(sqrt(3)-2)/4
  }
  __syncthreads();

  // if(i>(int) len - 9){
  //   printf("c2 X[%i]=%g\n",i,x_work[ish]);
  // }
  // __syncthreads();  

  //   if(threadIdx.x == 1){
  //     //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //     for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
  //   }
  // __syncthreads();

  //if(i==len-2) printf("X_CUDA2[%i] = %g\n",len-2,x_work[ish]);
  
  //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
    //  printf("herecyc3 thread%i\n",threadIdx.x);
    x_work[ish] = x_work[ish] - x_work[ish+3];
    //    x[i] = x[i] - x[i+3*skip];
    // s2[l] = s1[l] - d1[l+1]
  }
  __syncthreads();

  // if(i>len - 9){
  //   printf("c3 X[%i]=%g\n",i,x_work[ish]);
  // }
  // __syncthreads();

  //   if(threadIdx.x == 1){
  //     //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //     for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
  //   }
  // __syncthreads();

  //  if(i==len-2) printf("X_CUDA3[%i] = %g\n",len-2,x_work[ish]);

  //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
    //  printf("herecyc4 thread%i\n",threadIdx.x);
    x_work[ish] = x_work[ish]*Cl3;
    //    x[i] = x[i]*Cl3;
    // s[l] = s2[l]*(sqrt(3)-1)/sqrt(2)
    x_work[ish+1] = x_work[ish+1]*Cl4;
    //    x[i+skip] = x[i+skip]*Cl4;
    // d[l] = d1[l]*(sqrt(3)+1)/sqrt(2)
  }
  __syncthreads();

  // if(i==len-2) printf("X_CUDA4[%i] = %g\n",len-2,x_work[ish]);

  // if(i>len - 9){
  //   printf("c4 X[%i]=%g\n",i,x_work[ish]);
  // }
  // __syncthreads();

  //   if(threadIdx.x == 1){
  //     //  printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",levels,skip,len,skipwork);
  //     for(uint j=0; j<(BLOCK_SIZE2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);

  // }
  // __syncthreads();

  // Now transform level is done. We copy shared array x_work back into x
  //  if((i>=0)&&(i<len)){
  if((ish>=2) && (ish<(BLOCK_SIZE2<<1)-2) && (i<(int)len)){ //i>=0 as ish>=2
    //  printf("hereputback thread%i\n",threadIdx.x);
    x_out[i] = x_work[ish];
    x_out[i + skip] = x_work[ish+1];
    if(skip>1){
      x_out[i + (skip>>1)] = x_in[i + (skip>>1)];
      x_out[i + skip + (skip>>1)] = x_in[i + skip + (skip>>1)];
    }
  }
    
  __syncthreads();
  
}

  __global__ void Daub4_kernel_shared_b_io(real* x_in, real* x_out, const uint len, const uint skip){
  int i = (BLOCK_SECT2 * blockIdx.x + threadIdx.x -1)*skip<<1;
  // i = -2*skip, 0, 2*skip, ... , len-2*skip, len
  // for each block, we have, e.g. i = -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT2-2*skip, 2*BLOCK_SECT2*skip
  int ish = (threadIdx.x)<<1;
  // ish = 0, 2,... , 2*BLOCK_SECT2-2, 2*BLOCK_SECT2 2*BLOCK_SECT2+2
  __shared__ real x_work[BLOCK_SIZE2<<1];

  //###delete this when it's working!###
  // if(threadIdx.x == 0){
  //   for(uint j=0; j<(BLOCK_SIZE2<<1); j++) x_work[j] = 900+j;
  // }

  // First, we copy x into shared array x_work
  // NB - conditioning on threadIdx.x, not ish!
if(threadIdx.x == 0){ //ish == 0
    if(i<0){
      x_work[ish] = x_in[len - (skip<<1)];
      x_work[ish+1] = x_in[len - skip];
    }
    else if(i<(int)len){
      //      printf("here2 thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skip];
    }
  }
  
  if((threadIdx.x > 0) && (threadIdx.x < BLOCK_SECT2+1)){
    // needs to be conditional on i and BLOCK_SECT2
    if(i < (int)len){
      // printf("here3a thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skip];
    }
    else if(i==(int)len){
      // printf("here3b thread%i\n",threadIdx.x);
      x_work[ish] = x_in[0];
      x_work[ish+1] = x_in[skip];
    }
  }
  else if(threadIdx.x == BLOCK_SECT2+1){
    if(i<(int)len){
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skip];
    }
    else if(i==(int)len){
      x_work[ish] = x_in[0];
      x_work[ish+1] = x_in[skip];
    }
  }
  
  __syncthreads();

  //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
  //  printf("herecyc4 thread%i\n",threadIdx.x);
  x_work[ish] = x_work[ish]*Cl4;
  //    x[i] = x[i]*Cl4;
  // s[l] = s2[l]*(sqrt(3)+1)/sqrt(2)
  x_work[ish+1] = x_work[ish+1]*Cl3;
  //    x[i+skip] = x[i+skip]*Cl3;
  // d[l] = d1[l]*(sqrt(3)-1)/sqrt(2)
  //  }
  __syncthreads();
  
  //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
  if(ish<=(BLOCK_SECT2<<1)){
    //  printf("herecyc3 thread%i\n",threadIdx.x);
    x_work[ish] = x_work[ish] + x_work[ish+3];
    //    x[i] = x[i] + x[i+3*skip];
    // s2[l] = s1[l] + d1[l+1]
  }
  __syncthreads();

  //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(ish>=2){ // leaves 1 thread idle
    // printf("herecyc2 thread%i\n",threadIdx.x);
    x_work[ish+1] = x_work[ish+1] + x_work[ish]*Cl1 + x_work[ish-2]*Cl2;
    //    x[i+skip] = x[i+skip] + x[i]*Cl1 + x[i-2*skip]*Cl2;
    // d1[l] = x[2l+1] + s1[l]*sqrt(3)/4 + s1[l-1]*(sqrt(3)-2)/4
  }
  __syncthreads();

  //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle 
    //printf("herecyc1 thread%i\n",threadIdx.x);
    x_work[ish] = x_work[ish] - x_work[ish + 1]*Cl0;
    //    x[i] = x[i] - x[i+skip]*Cl0;
    // s1[l] = x[2l] - sqrt(3)*x[2l+1]
  }
  __syncthreads();
  
  // Now transform level is done. We copy shared array x_work back into x
  //  if((i>=0)&&(i<len)){
  if((ish>=2) && (ish<(BLOCK_SIZE2<<1)-2) && (i<len)){ //i>=0 as ish>=2
    //  printf("hereputback thread%i\n",threadIdx.x);
    x_out[i] = x_work[ish];
    x_out[i + skip] = x_work[ish+1];
  }

  __syncthreads();
}


// ########################################################################
// Now we do the same again but with a kernel that performs multiple layers
// of transform - like the Haar kernel
// -- input/output version --
// ########################################################################

/*
  Shared memory has following structure:

  _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ 
  |        |                                            |            |
  |_ _ _ _ |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ |_ _ _ _ _ _ |

  |<--4--->|<---------(2*BLOCK_SECT_ML2)--------------->|<---6------>|
  |<---------------(  2 * BLOCK_SIZE_ML2  )------------------------->|

  indices...
  |    2 3 |4 5 6 ....                                k |k1k2k3k4k5k6|
  used in 1st level of transform (where k=2*BLOCK_SECT_ML2+3) k1 is k+1 etc

  then indices...
  |0 1 2 3 |4 5 6 ....                                k |k1k2k3k4k5k6|
  used in 2nd level of transform

  First & last 4/6 coefficients contain shared memory boundary coefficients* for the transform levels.
  *shared memory boundary coefficients: for the first & last shared memory blocks, they hold periodic boundary coefficient points; for all other blocks, they hold the boundary coefficients of the previous/following memory block.

  The threads point to (via the variable ish):
  (ish is actual index in shared memory)
  (skipwork = 1)
  |0   1   |2   3 ...                               l   |l1  l2  l3  |
  where l is BLOCK_SECT_ML2+1
  (skipwork = 2)
  |0       |1     ...                           m       |m1      m2  |
  where m is floor[(BLOCK_SECT_ML2+1)/2]
*/

// ##########  To make this viable, we could use double pointers - then can ensure we return answer in correct vector. However, double pointers would have to be in main. HTen wrapper deals with what to return.   ###########

int Daub4CUDA_sh_ml2_io(real* x_d_in, real* x_d_out, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fDaub4CUDAsh_ml2_io(x_d_in, x_d_out,len,1,nlevels));
  case 0:
    return(bDaub4CUDAsh_ml2_io(x_d_in, x_d_out,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fDaub4CUDAsh_ml2_io(real* x_d_in, real* x_d_out, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock;
    int blocksPerGrid;
 
    uint levels=1; //leave at 1. This is initialisation for level variable!

    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    while((levels+1<=2)&&((skip<<(levels+1))<=(1 << nlevels))){ // levels+1<=k gives L, #levels to loop over
      // take skip to power levels+1 as filter of length 4
      levels+=1;
    }
    

    if (levels==1){   
      threadsPerBlock = BLOCK_SIZE2;
      blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;      
  
      Daub4_kernel_shared_f_io<<<blocksPerGrid, threadsPerBlock>>>(x_d_in, x_d_out,len,skip);

    }
    else{
      
      // printf("\nlevels=2");
      // printveccu<<<1,1>>>(bdrs,lenb);
      // cudaDeviceSynchronize();
      // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);
      
      threadsPerBlock = BLOCK_SIZE_ML2;
      blocksPerGrid =(len/(skip<<1) + BLOCK_SECT_ML2 - 1) / BLOCK_SECT_ML2;

      Daub4_kernel_shared_f_ml2_io<<<blocksPerGrid, threadsPerBlock>>>(x_d_in, x_d_out,len,skip);

    }

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fDaub4 sh io (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);

    // cudaDeviceSynchronize();
    
    return(fDaub4CUDAsh_ml2_io(x_d_out, x_d_in,len,skip<<levels,nlevels));

  }
  return(0);
}

int bDaub4CUDAsh_ml2_io(real* x_d_in, real* x_d_out, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock;
    int blocksPerGrid;

    uint levels=1; //leave at 1. This is initialisation for level variable!

    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    while((levels+1<=2)&&((skip>>levels)>0)){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }
    
    if (levels==1){
      // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);
      threadsPerBlock = BLOCK_SIZE2;
      blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
      Daub4_kernel_shared_b_io<<<blocksPerGrid, threadsPerBlock>>>(x_d_in, x_d_out,len,skip);

    }
    else{
      // printf("\nlevels=2");
      // printveccu<<<1,1>>>(bdrs,lenb);
      //cudaDeviceSynchronize();
      // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);
      threadsPerBlock = BLOCK_SIZE_ML2B;
      blocksPerGrid =(len/skip + BLOCK_SECT_ML2 - 1) / BLOCK_SECT_ML2;
      
      Daub4_kernel_shared_b_ml2_io<<<blocksPerGrid, threadsPerBlock>>>(x_d_in,x_d_out,len,skip);
    }
    
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in transform in bDaub4 sh io (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    
    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    
    //cudaDeviceSynchronize();

    return(bDaub4CUDAsh_ml2_io(x_d_out,x_d_in,len,skip>>levels));
    
  }
  return(0);
}


__global__ void Daub4_kernel_shared_f_ml2_io(real* x_in, real* x_out, const uint len, const uint skip){
  int i = (BLOCK_SECT_ML2 * blockIdx.x + threadIdx.x -2)*skip << 1;
  // i = -4*skip, -2*skip, 0, 2*skip, ... , len-2*skip, len, len +2
  // for each block, we have, e.g. i = -4*skip, -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT_ML2-2*skip, 2*BLOCK_SECT_ML2*skip, 2*skip*BLOCK_SECT_ML2+2*skip
  uint ish = (threadIdx.x)<<1;
  uint ishlast = (BLOCK_SIZE_ML2-1)<<1;
  uint li;
  // ish = 0, 2,... , 2*BLOCK_SECT_ML2-2, 2*BLOCK_SECT_ML2, 2*BLOCK_SECT_ML2+2, 2*BLOCK_SECT_ML2+4, 2*BLOCK_SECT_ML2+6, +8
  __shared__ real x_work[BLOCK_SIZE_ML2<<1];
  uint skipwork = 1;
  
  // printf("\nthreadIdx.x=%i,blockIdx.x=%i,i=%i,ish=%i\n",threadIdx.x,blockIdx.x,i,ish);

  // //###delete this when it's working!###
  // if(threadIdx.x == 0){
  //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) x_work[j] = 99900+j;
  // }

  // First, we copy x into shared array x_work
  // NB - conditioning on threadIdx.x, not ish!

  //## we copy values in for the first level of the transform

  //## bdrs contains 6 values at the start & end of each block => 12 in total
  //## We only use 4 at the start & 6 at the end.

  if(threadIdx.x <= 1){ //ish == 0 or 2
    if(i<0){
      x_work[ish] = x_in[len + skip*(ish-4)];
      x_work[ish+1] = x_in[len + skip*(ish-3)];
    }
    else if(i<(int)len){
      //      printf("here2 thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skip];
    }
  }
  
 //## this is the main section of the shared block
  if((threadIdx.x > 1) && (threadIdx.x < BLOCK_SECT_ML2+2)){
    // needs to be conditional on i and BLOCK_SECT
    if(i < len){
      // printf("here3a thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skip];
    }
    else if((i>=len)&&(i<=len+4*skip)){
      // ## then we are filling out boundary points in shared where length shared vector > len
      // printf("here3b thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i-len];
      x_work[ish+1] = x_in[i-len+skip];
    }
  }
  //## boundary coeffs
  else if(threadIdx.x >= BLOCK_SECT_ML2+2){
    if((i>=len)&&(i<=len+4*skip)){
      //  printf("here4 thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i-len];
      x_work[ish+1] = x_in[i-len+skip];
      //we pick up last 6 bdrs from each block
      //here, ish - (BLOCK_SECT_ML2<<1)  - 4 = (i-len)/skip
      //but the former should be faster to calculate!
    }
    else{
      if(i<len){
	//	printf("here5 thread%i\n",threadIdx.x);
	x_work[ish] = x_in[i];
	x_work[ish+1] = x_in[i + skip];
      }
    }
  }  
  __syncthreads();

  // if(threadIdx.x == 1){
  //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
  //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
  // }
  // __syncthreads();
  

  //we loop over a few levels...
  for(li = 0; li < 2; li++){
    
    //    if (ish < (BLOCK_SIZE2<<1)){ ## something like this!
    //## here we must restrict level 1 but not level 2

    //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast))||
	((li==1)&&(ish<=ishlast)) ){
      //printf("herecyc1 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish] + x_work[ish + skipwork]*Cl0;
      //    x[i] = x[i] + x[i+skip]*Cl0;
      // s1[l] = x[2l] + sqrt(3)*x[2l+1]
    }
    __syncthreads();

    // if(threadIdx.x == 1){
    //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
    //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
    // }
    // __syncthreads();

    
    //## different ifs for different levels.
    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    //   if( ((li==0)&&(ish>=4)&&(ish<(BLOCK_SIZE_ML2<<1)-2))||
    if( ((li==0)&&(ish>=2)&&(ish<=ishlast))||
	((li==1)&&(ish>=4)&&(ish<ishlast)) ){
      //    if(ish>=2){ // leaves 1 thread idle
      // printf("herecyc2 thread%i\n",threadIdx.x);
      x_work[ish+skipwork] = x_work[ish+skipwork] - x_work[ish]*Cl1 - x_work[ish-2*skipwork]*Cl2;
      //    x[i+skip] = x[i+skip] - x[i]*Cl1 - x[i-2*skip]*Cl2;
      // d1[l] = x[2l+1] - s1[l]*sqrt(3)/4 - s1[l-1]*(sqrt(3)-2)/4
    }
    __syncthreads();

    // if(threadIdx.x == 1){
    //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
    //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
    // }
    // __syncthreads();

  
    //## different ifs for different levels.
    //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    //if( ((li==0)&&(ish>=4)&&(ish<=(BLOCK_SIZE_ML2<<1)-4))||
    if( ((li==0)&&(ish<ishlast))||
	((li==1)&&(ish>=4)&&(ish<=ishlast-6)) ){
      //if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
      //  printf("herecyc3 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish] - x_work[ish+3*skipwork];
      //    x[i] = x[i] - x[i+3*skip];
      // s2[l] = s1[l] - d1[l+1]
    }
    __syncthreads();

    // if(threadIdx.x == 1){
    //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
    //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
    // }
    // __syncthreads();


    //## different ifs for different levels.
    //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    //    if( ((li==0)&&(ish>=4)&&(ish<=(BLOCK_SIZE_ML2<<1)-4))||
    if( ((li==0)&&(ish<=ishlast))||
	((li==1)&&(ish>=4)&&(ish<=ishlast-6)) ){
      // if((ish>=2)&&(ish<=(BLOCK_SECT2<<1))){ // leaves 2 threads idle
      //  printf("herecyc4 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish]*Cl3;
      //    x[i] = x[i]*Cl3;
      // s[l] = s2[l]*(sqrt(3)-1)/sqrt(2)
      x_work[ish+skipwork] = x_work[ish+skipwork]*Cl4;
      //    x[i+skip] = x[i+skip]*Cl4;
      // d[l] = d1[l]*(sqrt(3)+1)/sqrt(2)
    }
    __syncthreads();

    // if(threadIdx.x == 1){
    //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
    //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
    // }
    // __syncthreads();


    ish=ish<<1; skipwork=skipwork<<1;
    //__syncthreads();
    
  }
  

  // Now transform level is done. We copy shared array x_work back into x

  ish = (threadIdx.x)<<1;

  //  if((ish>=2) && (ish<(BLOCK_SIZE2<<1)-2) && (i<len)){ //i>=0 as ish>=2
  if( (ish>=4) && (ish<((BLOCK_SIZE_ML2<<1)-6)) && (i<len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x_out[i] = x_work[ish];
    x_out[i + skip] = x_work[ish+1];
    if(skip>2){
      x_out[i + (skip>>2)] = x_in[i + (skip>>2)];
      x_out[i + (skip>>1)] = x_in[i + (skip>>1)];
      x_out[i + (skip>>2) + (skip>>1)] = x_in[i + (skip>>2) + (skip>>1)];
      x_out[i + skip + (skip>>2)] = x_in[i + skip + (skip>>2)];
      x_out[i + skip + (skip>>1)] = x_in[i + skip + (skip>>1)];
      x_out[i + (skip<<1) - (skip>>2)] = x_in[i + (skip<<1) - (skip>>2)];
    }
  }

}

/*
  For backward transform, shared memory has slightly different (longer) structure:

   _ _ _ _ _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ 
  |                |                                            |            |
  |_ _ _ _ _ _ _ _ |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ |_ _ _ _ _ _ |

  |<--8----------->|<---------(2*BLOCK_SECT_ML2)--------------->|<---6------>|
  |<-----------------------(  2 * BLOCK_SIZE_ML2B )------------------------->|

*/

__global__ void Daub4_kernel_shared_b_ml2_io(real* x_in, real* x_out, const uint len, const uint skip){
  int i = (BLOCK_SECT_ML2 * blockIdx.x + threadIdx.x -4)*skip;
  // i = -2*skip, -skip,  0, ..., len-skip, len, len+1
  uint ish = (threadIdx.x)<<1;
  uint ishlast = min(len/(skip>>1)+10,(BLOCK_SIZE_ML2B-1)<<1); //size of shared vec x_work
  uint li;
  __shared__ real x_work[BLOCK_SIZE_ML2B<<1];
  uint skipwork = 2;

  uint skipbl = skip >>1; //actually, we need skip to be the skip of the 2nd layer of the transform, as that is the detail needed in xwork

  // //###delete this when it's working!###
  // if(threadIdx.x == 0){
  //   for(uint j=0; j<(BLOCK_SIZE_ML2B<<1); j++) x_work[j] = 99900+j;
  // }
  __syncthreads();

  //## we copy values in for the first level of the transform

  if(threadIdx.x <= 3){ //ish == 0, 2, 4 or 6
    if(i<0){
      x_work[ish] = x_in[len + skipbl*(ish-8)];
      x_work[ish+1] = x_in[len + skipbl*(ish-7)];
    }
    else if(i<(int) len){ //periodic boundary conditions
      // printf("here1 thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skipbl];
    }
  }
  
  //## this is the main section of the shared block
  if((threadIdx.x > 3) && (threadIdx.x < BLOCK_SECT_ML2+4)){
    // needs to be conditional on i and BLOCK_SECT
    if(i < len){
      // printf("here3a thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i];
      x_work[ish+1] = x_in[i + skipbl];
    }
    else if((i>=len)&&(i<=len+4*skipbl)){
      // ## then we are filling out boundary points in shared where length shared vector > len
      // printf("here3b thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i-len];
      x_work[ish+1] = x_in[i-len+skipbl];
    }
  }
  //## boundary coeffs
  else if(threadIdx.x >= BLOCK_SECT_ML2+4){
    if((i>=len)&&(i<=len+4*skipbl)){
      //  printf("here4 thread%i\n",threadIdx.x);
      x_work[ish] = x_in[i-len];
      x_work[ish+1] = x_in[i-len+skipbl];
    }
    else{
      if(i<len){
	//	printf("here5 thread%i\n",threadIdx.x);
	x_work[ish] = x_in[i];
	x_work[ish+1] = x_in[i + skipbl];
      }
    }
  }  
  __syncthreads();
  
  // if(threadIdx.x == 1){
  //   printf("\nLevels: %u, skip: %u, len: %u, skipwork: %u\n",2,skip,len,skipwork);
  //   for(uint j=0; j<(BLOCK_SIZE_ML2<<1); j++) printf("\nx_work.%i[%u] = %g",blockIdx.x,j,x_work[j]);
    
  // }
  // __syncthreads();

  ish=ish<<1;
  // __syncthreads();

  //we loop over a few levels...
  for(li = 0; li < 2; li++){
    
    //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    if( ((li==0)&&(ish<ishlast))||
	((li==1)&&(ish<ishlast)) ){
      //  printf("herecyc4 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish]*Cl4;
      //    x[i] = x[i]*Cl4;
      // s[l] = s2[l]*(sqrt(3)+1)/sqrt(2)
      x_work[ish+skipwork] = x_work[ish+skipwork]*Cl3;
      //    x[i+skip] = x[i+skip]*Cl3;
      // d[l] = d1[l]*(sqrt(3)-1)/sqrt(2)
    }
    __syncthreads();
    
    //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    if( ((li==0)&&(ish<ishlast))||
	((li==1)&&(ish<ishlast)) ){
      //  printf("herecyc3 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish] + x_work[ish+3*skipwork];
      //    x[i] = x[i] + x[i+3*skip];
      // s2[l] = s1[l] + d1[l+1]
    }
    __syncthreads();
    
    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for the first level, we need to ensure all boundary coeffs are kept updated
    if( ((li==0)&&(ish>=2)&&(ish<ishlast))||
	((li==1)&&(ish>=2)&&(ish<ishlast)) ){
      // printf("herecyc2 thread%i\n",threadIdx.x);
      x_work[ish+skipwork] = x_work[ish+skipwork] + x_work[ish]*Cl1 + x_work[ish-2*skipwork]*Cl2;
      //    x[i+skip] = x[i+skip] + x[i]*Cl1 + x[i-2*skip]*Cl2;
      // d1[l] = x[2l+1] + s1[l]*sqrt(3)/4 + s1[l-1]*(sqrt(3)-2)/4
    }
    __syncthreads();
  
    //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish>=2)&&(ish<ishlast))||
	((li==1)&&(ish>=2)&&(ish<ishlast)) ){
      //printf("herecyc1 thread%i\n",threadIdx.x);
      x_work[ish] = x_work[ish] - x_work[ish + skipwork]*Cl0;
      //    x[i] = x[i] - x[i+skip]*Cl0;
      // s1[l] = x[2l] - sqrt(3)*x[2l+1]
    }
    __syncthreads();

    ish=ish>>1; skipwork=skipwork>>1;
    //__syncthreads();
   
  } 
  
  // Now transform level is done. We copy shared array x_work back into x
  
  ish = (threadIdx.x)<<1;
  
  if( (ish>=8) && (ish<((BLOCK_SIZE_ML2B<<1)-6)) && (i<len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x_out[i] = x_work[ish];
    x_out[i + skipbl] = x_work[ish+1];
  }
  
  // __syncthreads();

}