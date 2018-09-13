#include "la8cuda.cuh"
#include "la8coeffs.h"

#define BLOCK_SIZE2 263 // (8 down, 6 up)/2
#define BLOCK_SECT2 256

#define BLOCK_SIZE_ML2 278 //(24 down, 20 up)/2
#define BLOCK_SIZE_ML2B 274 //(20 down, 16 up)/2
// block sect remains the same.
#define BLOCK_SECT_ML2 256

/***********************************************************
Lifted Least Asymmetric 8 code - in CUDA! With shared memory
***********************************************************/


int LA8CUDA_sh(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=8;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, in the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fLA8CUDAsh(x_d,len,1,nlevels));
  case 0:
    return(bLA8CUDAsh(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fLA8CUDAsh(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE2;
    int blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
    int res;

    uint k = 8; // max(above,below) # border coeffs obtained
    real* bdrs; // vector of boundary points - ensures independence of loops
    uint lenb = max((len*k)/(skip*BLOCK_SECT2),2*k); // length of bdrs vector
    int tPB_bd = BS_BD;
    int bPG_bd = max(((lenb/(2*k)) + BS_BD - 1) / BS_BD,1);
    cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
    get_bdrs_sh_k<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb,k,BLOCK_SECT2); //we copy the boundary points into a vector
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

    LA8_kernel_shared_f<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in LA8 sh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    cudaFree(bdrs);

    res=fLA8CUDAsh(x_d,len,skip<<1,nlevels);
    cudaDeviceSynchronize();
    return(res);
  }
    
  return(0);
}

int bLA8CUDAsh(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE2;
    int blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
    int res;

    uint k = 8; // max(above,below) # border coeffs obtained
    real* bdrs; // vector of boundary points - ensures independence of loops
    uint lenb = max((len*k)/(skip*BLOCK_SECT2),2*k); // length of bdrs vector
    int tPB_bd = BS_BD;
    int bPG_bd = max(((lenb/(2*k)) + BS_BD - 1) / BS_BD,1);
    cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
    get_bdrs_sh_k<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb,k,BLOCK_SECT2); //we copy the boundary points into a vector
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
    
    LA8_kernel_shared_b<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);
    
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in LA8 sh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    cudaFree(bdrs);

    res=bLA8CUDAsh(x_d,len,skip>>1);
    cudaDeviceSynchronize();
    return(res);
  }
  return(0);
}


/*
  Shared memory has following structure:

   _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  |           |                                           |    |
  |_ _ _ _ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|_ _ |

  |<----8---->|<---------(2*BLOCK_SECT2  )--------------->|--6-|
  |<----------------(  2 * BLOCK_SIZE2    )------------------->|

  Where of the 8 spaces below, we need 7; of the 6 spaces above, we need 5.
  But our memory structure dictates that we use even values of 'i'

*/

__global__ void LA8_kernel_shared_f(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT2 * blockIdx.x + threadIdx.x -4)*skip<<1;
  // i = -8*skip, -6*skip, -4*skip, -2*skip 0, 2*skip, ... , len-2*skip, len, len + 2*skip, len + 4*skip, len + 6*skip
  // for each block, we have, e.g. i = -8*skip, -6*skip, -4*skip, -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT2-2*skip, 2*skip*BLOCK_SECT2, 2*skip*BLOCK_SECT2+2*skip, 2*skip*BLOCK_SECT2+4*skip, 2*skip*BLOCK_SECT2+4*skip
  int ish = (threadIdx.x)<<1;
  // ish = 0, 2,... , 2*BLOCK_SIZE2-2, 2*BLOCK_SIZE2
  __shared__ real x_work[BLOCK_SIZE2<<1];

  // real switchsd;
  
  // printf("\nthreadIdx.x=%i,blockIdx.x=%i,i=%i,ish=%i\n",threadIdx.x,blockIdx.x,i,ish);

  // // First, we copy x into shared array x_work
  // // NB - conditioning on threadIdx.x, not ish!
  // if(threadIdx.x < 6){ //ish == 0, 2, 4, 6, 8, 10
  //   if(i<(int)len){
  //     // we are filling the shared block with lower boundary points
  //     //      printf("here2 thread%i\n",threadIdx.x);
  //     // x_work[ish] = x[i];
  //     // x_work[ish+1] = x[i + skip];
  //     x_work[ish] = bdrs[ish + blockIdx.x*12];
  //     x_work[ish+1] = bdrs[1 + ish + blockIdx.x*12];
  //   }
  // }
  
  // if((threadIdx.x >= 6) && (threadIdx.x < BLOCK_SECT2+6)){
  //   // needs to be conditional on i and BLOCK_SECT2
  //   if(i < (int)len){
  //     // we fill the central block of shared memory (no boundary coeffs)
  //     // printf("here3a thread%i\n",threadIdx.x);
  //     x_work[ish] = x[i];
  //     x_work[ish+1] = x[i + skip];
  //   }

  //   else if(i==(int)len){
  //     // this happens when len < BLOCK_SECT2
  //     // we have to deal with upper boundary points
  //     // printf("here3b thread%i\n",threadIdx.x);
  //     // x_work[ish] = x[0];
  //     // x_work[ish+1] = x[skip];
  //     x_work[ish] = bdrs[6+(blockIdx.x*12)];
  //     x_work[ish+1] = bdrs[7+(blockIdx.x*12)];
  //   }
  // }
  // else if(threadIdx.x == BLOCK_SECT2+4){
  //   if(i<=(int)len){
  //     x_work[ish] = bdrs[6+(blockIdx.x*12)];
  //     x_work[ish+1] = bdrs[7+(blockIdx.x*12)];
  //   }
  // }
  
  // x_work[ish] = get_wvt_shared(x,bdrs,len,skip,i,ish,threadIdx.x, blockIdx.x,1);
  // x_work[ish+1] = get_wvt_shared(x,bdrs,len,skip,i,ish,threadIdx.x, blockIdx.x,0);
 
  // x_work[ish] = get_wvt_shared_gen(x,bdrs,len,skip,i,ish,8,6,BLOCK_SECT2,1);
  // x_work[ish+1] = get_wvt_shared_gen(x,bdrs,len,skip,i,ish,8,6,BLOCK_SECT2,0);
 
  write_wvt_shared_gen(x,bdrs,len,skip,i,ish,8,6,BLOCK_SECT2,x_work);
 
  __syncthreads();

  // in the lifting cycles below, we keep some threads idle so that we only update
  // the intermediate coefficients that we need in the subsequent steps


  //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1+12))){ //keep 2 threads idle
    //printf("herecyc1 thread%i\n",threadIdx.x);
    lift_cyc_1(x_work,ish,1,FWD);
    // x_work[ish] = x_work[ish] + x_work[ish-1]*CL0 + x_work[ish+1]*CL1;
    // d1[l] = x[2l+1] + q11*x[2l] + q12*x[2l+2]
  }
  __syncthreads();

    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<=(BLOCK_SECT2<<1+12))){ //keep 2 threads idle
    // printf("herecyc2 thread%i\n",threadIdx.x);
    lift_cyc_2(x_work,ish,1,FWD);
    // x_work[ish-1] = x_work[ish-1] + x_work[ish]*CL2 + x_work[ish+2]*CL3;
    // s1[l] = x[2l] + q21*d1[l] + q22*d1[l+1]
  }
  __syncthreads();


  //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<(BLOCK_SECT2<<1+8))){ //keep 3 threads idle
    //  printf("herecyc3 thread%i\n",threadIdx.x);
    lift_cyc_3(x_work,ish,1,FWD);
    // x_work[ish] = x_work[ish] + x_work[ish-1]*CL4 + x_work[ish+1]*CL5;
    // d2[l] = d1[l] + q31*s1[l] + q32*s1[l+1]
  }
  __syncthreads();


  //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=4)&&(ish<(BLOCK_SECT2<<1+8))){ //keep 4 threads idle
    lift_cyc_4(x_work,ish,1,FWD);
    // x_work[ish-1] = x_work[ish-1] + x_work[ish-2]*CL6 + x_work[ish]*CL7;
    // s2[l] = s1[l] + q41*d2[l-1] + q42*d2[l]
  }
  __syncthreads();

  
  //5th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=8)&&(ish<(BLOCK_SECT2<<1+8))){ // keep 6 threads idle
    lift_cyc_5(x_work,ish,1,FWD);
    // x_work[ish] = x_work[ish] + x_work[ish-5]*CL8 + x_work[ish-3]*CL9 + x_work[ish-1]*CL10;
    // d3[l] = d2[l] + s1*K^2*s2[l-2] + s2*K^2*s2[l-1] + s3*K^2*s2[l]
  }
  __syncthreads();


  // //6th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ - done below
  // if((ish>=8)&&(ish<(BLOCK_SECT2<<1+8))){
  //   switchsd = x_work[ish]*CL12;
  //   //s3[l] = (K)*s2[l]
  //   x_work[ish] = x_work[ish+1]*CL11;
  //   // d4[l] = (1/K)*d3[l]
  //   x_work[ish+1] = switchsd;    
  // }
  // __syncthreads();
  
  
  // We do last lifting cycle at the same time as writing back to global memory.
  // Involves a switch in coefficients because of the derived lifting algo.
  if((ish>=8) && (ish<(BLOCK_SECT2<<1)+8) && (i<(int)len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish+1]*CL11;
    x[i + skip] = x_work[ish]*CL12;
  }

  __syncthreads();


}

// same memory diagram as for the forward transform

__global__ void LA8_kernel_shared_b(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT2 * blockIdx.x + threadIdx.x -4)*skip<<1;
  // i = -8*skip, -6*skip, -4*skip, -2*skip 0, 2*skip, ... , len-2*skip, len, len + 2*skip, len + 4*skip, len + 6*skip
  // for each block, we have, e.g. i = -8*skip, -6*skip, -4*skip, -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT2-2*skip, 2*skip*BLOCK_SECT2, 2*skip*BLOCK_SECT2+2*skip, 2*skip*BLOCK_SECT2+4*skip, 2*skip*BLOCK_SECT2+4*skip
  int ish = (threadIdx.x)<<1;
  // ish = 0, 2,... , 2*BLOCK_SIZE2-2, 2*BLOCK_SIZE2
  __shared__ real x_work[BLOCK_SIZE2<<1];
  // printf("\nthreadIdx.x=%i,blockIdx.x=%i,i=%i,ish=%i\n",threadIdx.x,blockIdx.x,i,ish);
 
  x_work[ish] = get_wvt_shared_gen(x,bdrs,len,skip,i,ish,8,6,BLOCK_SECT2,0) * CL11;
  x_work[ish+1] = get_wvt_shared_gen(x,bdrs,len,skip,i,ish,8,6,BLOCK_SECT2,1) * CL12;
  // above, we do the last lifting cycle at the same time as reading the coefficients
  // we have switched the odd/even coefficients above because our derived lifting algo
  // requires that
 
  __syncthreads();

  // in the lifting cycles below, we keep some threads idle so that we only update
  // the intermediate coefficients that we need in the subsequent steps
  
  //5th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=6)&&(ish<(BLOCK_SECT2<<1+14))){ // keep 3 threads idle
    lift_cyc_5(x_work,ish,1,BWD);
    // x_work[ish] = x_work[ish] - x_work[ish-5]*CL8 - x_work[ish-3]*CL9 - x_work[ish-1]*CL10;
    // d3[l] = d2[l] - s1*K^2*s2[l-2] - s2*K^2*s2[l-1] - s3*K^2*s2[l]
  }
  __syncthreads();


  //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=6)&&(ish<(BLOCK_SECT2<<1+14))){ //keep 4 threads idle
    lift_cyc_4(x_work,ish,1,BWD);    
    // x_work[ish-1] = x_work[ish-1] - x_work[ish-2]*CL6 - x_work[ish]*CL7;
    // s2[l] = s1[l] - q41*d2[l-1] - q42*d2[l]
  }
  __syncthreads();


  //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=8)&&(ish<(BLOCK_SECT2<<1+12))){ //keep 5 threads idle
    //  printf("herecyc3 thread%i\n",threadIdx.x);
    lift_cyc_3(x_work,ish,1,BWD);
    // x_work[ish] = x_work[ish] - x_work[ish-1]*CL4 - x_work[ish+1]*CL5;
    // d2[l] = d1[l] - q31*s1[l] - q32*s1[l+1]
  }
  __syncthreads();


    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=6)&&(ish< (BLOCK_SECT2<<1+12))){ //keep 6 threads idle
    // printf("herecyc2 thread%i\n",threadIdx.x);
    lift_cyc_2(x_work,ish,1,BWD);
    // x_work[ish-1] = x_work[ish-1] - x_work[ish]*CL2 - x_work[ish+2]*CL3;
    // s1[l] = x[2l] - q21*d1[l] - q22*d1[l+1]
  }
  __syncthreads();


  //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=8)&&(ish<(BLOCK_SECT2<<1+10))){ //keep 7 threads idle
    //printf("herecyc1 thread%i\n",threadIdx.x);
    lift_cyc_1(x_work,ish,1,BWD);
    // x_work[ish] = x_work[ish] - x_work[ish-1]*CL0 - x_work[ish+1]*CL1;
    // d1[l] = x[2l+1] - q11*x[2l] - q12*x[2l+2]
  }
  __syncthreads();


  // We write back to global memory.
  if((ish>=8) && (ish<(BLOCK_SECT2<<1)+8) && (i<(int)len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skip] = x_work[ish+1];
  }

  __syncthreads();


}


// ########################################################################
// Now we have a kernel that performs 2 levels of transform
// ########################################################################

/*
  Shared memory has following, new, structure:

   _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ 
  |        |                                            |            |
  |_ _ _ _ |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ |_ _ _ _ _ _ |

  |<--24-->|<---------(2*BLOCK_SECT_ML2)--------------->|<---20----->|
  |<---------------(  2 * BLOCK_SIZE_ML2  )------------------------->|

  indices...
  1, 2, 3, ..., k, k1, ..., k17
  used in 1st level of transform (where k=2*BLOCK_SECT_ML2+22) k1 is k+1 etc

  then indices...
  8, 9, 10, ..., k, k1, ..., k11
  used in 2nd level of transform

  First & last 4/6 coefficients contain shared memory boundary coefficients* for the transform levels.
  *shared memory boundary coefficients: for the first & last shared memory blocks, they hold periodic boundary coefficient points; for all other blocks, they hold the boundary coefficients of the previous/following memory block.

  The threads point to (via the variable ish):
  (ish is actual index in shared memory)
  (skipwork = 1)
  |0 1  ... 23 |24    25 ...                               l   |l1  l2  ... l20 |
  where l is BLOCK_SECT_ML2+1
  (skipwork = 2)
  |0  ...    11 |12     ...                           m       |m1  ...    m10  |
  where m is floor[(BLOCK_SECT_ML2+1)/2]
*/

// above is probably not quite right! :D


int LA8CUDA_sh_ml2(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=8;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fLA8CUDAsh_ml2(x_d,len,1,nlevels));
  case 0:
    return(bLA8CUDAsh_ml2(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fLA8CUDAsh_ml2(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock;
    int blocksPerGrid;
 
    uint levels=1; //leave at 1. This is initialisation for level variable!

    uint k1 = 8; //# border coeffs needed for single level kernel
    uint k2 = 24; //# coeffs needed for 2 level kernel

    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    while((levels+1<=2)&&((skip<<(levels+1))<=(1 << nlevels))&&(len/skip>k2)){ // levels+1<=k gives L, #levels to loop over
      
      levels+=1;
    }
    

    if (levels==1){
      // deal with bdrs
      uint k = k1; // max(above,below) # border coeffs obtained
      real* bdrs; // vector of boundary points - ensures independence of loops
      uint lenb = max((len*k)/(skip*BLOCK_SECT2),2*k); // length of bdrs vector
      int tPB_bd = BS_BD;
      int bPG_bd = max(((lenb/(2*k)) + BS_BD - 1) / BS_BD,1);
      cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
      get_bdrs_sh_k<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb,k,BLOCK_SECT2); //we copy the boundary points into a vector


      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaDeviceSynchronize();
      
      threadsPerBlock = BLOCK_SIZE2;
      blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;      

      LA8_kernel_shared_f<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

      cudaFree(bdrs);

    }
    else{
      // deal with bdrs
      real* bdrs; // vector of boundary points - ensures independence of loops
      uint k = k2;
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
                  
      LA8_kernel_shared_f_ml2<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);
      cudaFree(bdrs);

    }

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fLA8 MLsh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);

    cudaDeviceSynchronize();
    
    return(fLA8CUDAsh_ml2(x_d,len,skip<<levels,nlevels));

  }
  return(0);
}


int bLA8CUDAsh_ml2(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock;
    int blocksPerGrid;
    real* bdrs;
    
    uint levels=1; //leave at 1. This is initialisation for level variable!

    uint k1 = 8; //# border coeffs needed for single level kernel
    uint k2 = 20; //# coeffs needed for 2 level kernel

    // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);

    while((levels+1<=2)&&((skip>>levels)>0)&&(len/skip>k2)){ // levels+1<=k gives L, #levels to loop over
      levels+=1;
    }

    if (levels==1){
      // deal with bdrs
      uint k = k1; // max(above,below) # border coeffs obtained
      real* bdrs; // vector of boundary points - ensures independence of loops
      uint lenb = max((len*k)/(skip*BLOCK_SECT2),2*k); // length of bdrs vector
      int tPB_bd = BS_BD;
      int bPG_bd = max(((lenb/(2*k)) + BS_BD - 1) / BS_BD,1);
      cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
      get_bdrs_sh_k<<<bPG_bd, tPB_bd>>>(x_d,len,skip,bdrs,lenb,k,BLOCK_SECT2); //we copy the boundary points into a vector


      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaDeviceSynchronize();
      
      threadsPerBlock = BLOCK_SIZE2;
      blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;      

      LA8_kernel_shared_b<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

      cudaFree(bdrs);

    }
    else{
      // deal with bdrs
      real* bdrs; // vector of boundary points - ensures independence of loops
      uint k = k2;
      uint lenb = max((len*k*2)/(skip*BLOCK_SECT_ML2),2*k); // length of bdrs vector
      int tPB_bd = BS_BD;
      int bPG_bd = max(((lenb/(2*k)) + BS_BD - 1) / BS_BD,1);
      cudaMalloc((void **)&bdrs,lenb*sizeof(real));      
      get_bdrs_sh_k<<<bPG_bd, tPB_bd>>>(x_d,len,skip/2,bdrs,lenb,k,BLOCK_SECT_ML2); //we copy the boundary points into a vector
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform in get boundaries sh (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      cudaDeviceSynchronize();
      
      threadsPerBlock = BLOCK_SIZE_ML2B;
      blocksPerGrid =(len/skip + BLOCK_SECT_ML2 - 1) / BLOCK_SECT_ML2;

      // printf("\nlevels=2");
      // printveccu<<<1,1>>>(bdrs,lenb);
      // cudaDeviceSynchronize();
      // printf("\n### threadsperblock = %i, blockspergrid = %i ####\n",threadsPerBlock,blocksPerGrid);
                  
      LA8_kernel_shared_b_ml2<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);
      cudaFree(bdrs);

    }

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bLA8 MLsh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);

    cudaDeviceSynchronize();
    
    return(bLA8CUDAsh_ml2(x_d,len,skip>>levels));

  }

  return(0);
}

__global__ void LA8_kernel_shared_f_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT_ML2 * blockIdx.x + threadIdx.x -12)*skip << 1;
  // i = -24*skip,..., -2*skip, 0, 2*skip, ... , len-2*skip, len, len +2, ... , len + 16*skip, len + 18*skip
  // for each block, we have, e.g. i = -24*skip, ..., -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT_ML2-2*skip, 2*BLOCK_SECT_ML2*skip, 2*skip*BLOCK_SECT_ML2+2*skip, ..., 2*skip*BLOCK_SECT_ML2+20*skip
  uint ish = (threadIdx.x)<<1;
  uint ishlast = (BLOCK_SIZE_ML2-1)<<1;
  uint li;
  // ish = 0, 2,... , 2*BLOCK_SECT_ML2-2, 2*BLOCK_SECT_ML2, 2*BLOCK_SECT_ML2+2, 2*BLOCK_SECT_ML2+4, 2*BLOCK_SECT_ML2+6, +8, ... , +38
  __shared__ real x_work[BLOCK_SIZE_ML2<<1];
  uint skipwork = 1;

  write_wvt_shared_gen(x,bdrs,len,skip,i,ish,24,20,BLOCK_SECT2,x_work);
  __syncthreads();

  for(li = 0; li < 2; li++){
    
    //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-2)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-8)&&(ish>=12)) ){
      lift_cyc_1(x_work,ish,skipwork,FWD);
    }
    __syncthreads();

    
    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-2)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-8)&&(ish>=12)) ){
      lift_cyc_2(x_work,ish,skipwork,FWD);
    }
    __syncthreads();
    
    
    //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-6)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-14)&&(ish>=12)) ){
      lift_cyc_3(x_work,ish,skipwork,FWD);
    }
    __syncthreads();
    

    //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-6)&&(ish>=6)) ||
	((li==1)&&(ish<=ishlast-16)&&(ish>=16)) ){
      lift_cyc_4(x_work,ish,skipwork,FWD);
    }
    __syncthreads();
    
    
    //5th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-8)&&(ish>=10)) ||
	((li==1)&&(ish<=ishlast-20)&&(ish>=24)) ){
      lift_cyc_5(x_work,ish,skipwork,FWD);
    }
    __syncthreads();

    
    //6th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-8)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-20)&&(ish>=22)) ){
      lift_cyc_6(x_work,ish,skipwork,FWD);
    }
    __syncthreads();
    
    
    if(li==0) ish=ish<<1; skipwork=skipwork<<1;

  }

  
  // Now transform level is done. We copy shared array x_work back into x

  ish = (threadIdx.x)<<1;

  if( (ish>=24) && (ish<2*BLOCK_SECT_ML2+24) && (i<len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skip] = x_work[ish+1];
  }
  
  
}


__global__ void LA8_kernel_shared_b_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT_ML2 * blockIdx.x + threadIdx.x -10)*skip;
  // i = -24*skip,..., -2*skip, 0, 2*skip, ... , len-2*skip, len, len +2, ... , len + 16*skip, len + 18*skip
  // for each block, we have, e.g. i = -24*skip, ..., -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT_ML2-2*skip, 2*BLOCK_SECT_ML2*skip, 2*skip*BLOCK_SECT_ML2+2*skip, ..., 2*skip*BLOCK_SECT_ML2+20*skip
  uint skipbl = skip/2; // skip value for second layer. Used in filling shared mem & filling global memory vector.
  uint ish = (threadIdx.x)<<1;
  uint ishlast = min(len/skipbl+20+16,(BLOCK_SIZE_ML2-1)<<1); //size of shared vec x_work
  uint li;

  // ish = 0, 2,... , 2*BLOCK_SECT_ML2-2, 2*BLOCK_SECT_ML2, 2*BLOCK_SECT_ML2+2, 2*BLOCK_SECT_ML2+4, 2*BLOCK_SECT_ML2+6, +8, ... , +38
  __shared__ real x_work[BLOCK_SIZE_ML2<<1];
  uint skipwork = 2;

  write_wvt_shared_gen(x,bdrs,len,skipbl,i,ish,20,16,BLOCK_SECT2,x_work);
  __syncthreads();

  ish=ish<<1; 

  for(li = 0; li < 2; li++){
   
    //6th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=0)) ||
	((li==1)&&(ish<=ishlast-10)&&(ish>=14)) ){
      lift_cyc_6(x_work,ish,skipwork,BWD);
    }
    __syncthreads();


    //5th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=12)) ||
	((li==1)&&(ish<=ishlast-10)&&(ish>=14)) ){
      lift_cyc_5(x_work,ish,skipwork,BWD);
    }
    __syncthreads();

    
    //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=12)) ||
	((li==1)&&(ish<=ishlast-12)&&(ish>=18)) ){
      lift_cyc_4(x_work,ish,skipwork,BWD);
    }
    __syncthreads();
    
    
    //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-6)&&(ish>=16)) ||
	((li==1)&&(ish<=ishlast-12)&&(ish>=18)) ){
      lift_cyc_3(x_work,ish,skipwork,BWD);
    }
    __syncthreads();
    

    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-8)&&(ish>=16)) ||
	((li==1)&&(ish<=ishlast-12)&&(ish>=20)) ){
      lift_cyc_2(x_work,ish,skipwork,BWD);
    }
    __syncthreads();
           
    
    //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-10)&&(ish>=16)) ||
	((li==1)&&(ish<=ishlast-16)&&(ish>=20)) ){
      lift_cyc_1(x_work,ish,skipwork,BWD);
    }
    __syncthreads();

    
    if(li==0) ish=ish>>1; skipwork=skipwork>>1;

  }

  
  // Now transform level is done. We copy shared array x_work back into x

  ish = (threadIdx.x)<<1;

  if( (ish>=20) && (ish<2*BLOCK_SECT_ML2+20) && (i<len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skipbl] = x_work[ish+1];
  }
  
  
  

}


static __device__ void lift_cyc_1(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish] = xsh[ish] + xsh[ish-sk]*CL0 + xsh[ish+sk]*CL1;
  else xsh[ish] = xsh[ish] - xsh[ish-sk]*CL0 - xsh[ish+sk]*CL1;
  // d1[l] = x[2l+1] + q11*x[2l] + q12*x[2l+2]
}

static __device__ void lift_cyc_2(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish-sk] = xsh[ish-sk] + xsh[ish]*CL2 + xsh[ish+2*sk]*CL3;
  else xsh[ish-sk] = xsh[ish-sk] - xsh[ish]*CL2 - xsh[ish+2*sk]*CL3;
  // s1[l] = x[2l] + q21*d1[l] + q22*d1[l+1]
}

static __device__ void lift_cyc_3(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish] = xsh[ish] + xsh[ish-sk]*CL4 + xsh[ish+sk]*CL5;
  else xsh[ish] = xsh[ish] - xsh[ish-sk]*CL4 - xsh[ish+sk]*CL5;
  // d2[l] = d1[l] + q31*s1[l] + q32*s1[l+1]
}

static __device__ void lift_cyc_4(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish-sk] = xsh[ish-sk] + xsh[ish-2*sk]*CL6 + xsh[ish]*CL7;
  else xsh[ish-sk] = xsh[ish-sk] - xsh[ish-2*sk]*CL6 - xsh[ish]*CL7;
  // s2[l] = s1[l] + q41*d2[l-1] + q42*d2[l]
}

static __device__ void lift_cyc_5(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish] = xsh[ish] + xsh[ish-5*sk]*CL8 + xsh[ish-3*sk]*CL9 + xsh[ish-sk]*CL10;
  else xsh[ish] = xsh[ish] - xsh[ish-5*sk]*CL8 - xsh[ish-3*sk]*CL9 - xsh[ish-sk]*CL10;
  // d3[l] = d2[l] + s1*K^2*s2[l-2] + s2*K^2*s2[l-1] + s3*K^2*s2[l]
}

static __device__ void lift_cyc_6(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD){
    real switchsd = xsh[ish]*CL12;
    xsh[ish] = xsh[ish+sk]*CL11;
    xsh[ish+sk] = switchsd;
  }
  else{
    real switchsd = xsh[ish]*CL12;
    xsh[ish] = xsh[ish+sk]*CL11;
    xsh[ish+sk] = switchsd;
  } 
  //s3[l] = (K)*s2[l]
  // d4[l] = (1/K)*d3[l]
}

// // no longer used!
// __device__ double get_wvt_shared(real* x, real* bdrs, const uint len, const uint skip, const int i, const int ish, const short isskip){

//   // First, we copy x into shared array x_work
//   // NB - conditioning on threadIdx.x, not ish!
//   if(threadIdx.x < 6){ //ish == 0, 2, 4, 6, 8, 10
//     if(i<(int)len){
//       // we are filling the shared block with lower boundary points
//       //      printf("here2 thread%i\n",threadIdx.x);
//       // x_work[ish] = x[i];
//       // x_work[ish+1] = x[i + skip];
//       if(isskip) return(bdrs[ish + blockIdx.x*12]);
//       else return(bdrs[1 + ish + blockIdx.x*12]);
//     }
//   }
  
//   if((threadIdx.x >= 6) && (threadIdx.x < BLOCK_SECT2+6)){
//     // needs to be conditional on i and BLOCK_SECT2
//     if(i < (int)len){
//       // we fill the central block of shared memory (no boundary coeffs)
//       // printf("here3a thread%i\n",threadIdx.x);
//       if(isskip) return(x[i]);
//       else return(x[i + skip]);
//     }
    
//     else if(i==(int)len){
//       // this happens when len < BLOCK_SECT2
//       // we have to deal with upper boundary points
//       // printf("here3b thread%i\n",threadIdx.x);
//       // x_work[ish] = x[0];
//       // x_work[ish+1] = x[skip];
//       if(isskip) return(bdrs[6+(blockIdx.x*12)]);
//       else return(bdrs[7+(blockIdx.x*12)]);
//     }
//   }
//   else if(threadIdx.x == BLOCK_SECT2+4){
//     if(i<=(int)len){
//       if(isskip) return(bdrs[6+(blockIdx.x*12)]);
//       else return(bdrs[7+(blockIdx.x*12)]);
//     }
//   }
//   return(9999);
// }



int LA8CUDA_sh_ml2_streams(real* x_h, real* x_d, uint len, short int sense, uint nlevels, cudaStream_t stream){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=8;
  uint ret;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  cudaMemcpyAsync(x_d,x_h,len*sizeof(real),HTD,stream);
  switch(sense){
  case 1:
    ret = fLA8CUDAsh_ml2(x_d,len,1,nlevels);
    break;
  case 0:
    ret = bLA8CUDAsh_ml2(x_d,len,1<<(nlevels-1));
    break;
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
  cudaMemcpyAsync(x_h,x_d,len*sizeof(real),DTH,stream);
  // we copy x_d back into x_h
  // we have to do this after the DWT, as the transform is in-place
  return(ret);
}
