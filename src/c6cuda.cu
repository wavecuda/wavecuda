#include "c6cuda.cuh"
#include "c6coeffs.h"

#define BLOCK_SIZE2 258 // (2 down, 2 up)/2
#define BLOCK_SECT2 256

#define BLOCK_SIZE_ML2 264 //(8 down, 8 up)/2
#define BLOCK_SIZE_ML2B 264 //(8 down, 8 up)/2
// block sect remains the same.
#define BLOCK_SECT_ML2 256

/***********************************************************
Lifted Coiflet 6 code - in CUDA! With shared memory
***********************************************************/


int C6CUDA_sh(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=6;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, in the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fC6CUDAsh(x_d,len,1,nlevels));
  case 0:
    return(bC6CUDAsh(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fC6CUDAsh(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE2;
    int blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
    int res;

    uint k = 2; // max(above,below) # border coeffs obtained
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

    C6_kernel_shared_f<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in C6 sh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    cudaFree(bdrs);

    res=fC6CUDAsh(x_d,len,skip<<1,nlevels);
    cudaDeviceSynchronize();
    return(res);
  }
    
  return(0);
}

int bC6CUDAsh(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock = BLOCK_SIZE2;
    int blocksPerGrid =(len/(skip<<1) + BLOCK_SECT2 - 1) / BLOCK_SECT2;
    int res;

    uint k = 2; // max(above,below) # border coeffs obtained
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
    
    C6_kernel_shared_b<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);
    
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in C6 sh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);
    // cudaDeviceSynchronize();
    
    cudaFree(bdrs);

    res=bC6CUDAsh(x_d,len,skip>>1);
    cudaDeviceSynchronize();
    return(res);
  }
  return(0);
}


/*
  Shared memory has following structure:

    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  |    |                                           |    |
  | _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|_ _ |

  |--2-|<---------(2*BLOCK_SECT2  )--------------->|--2-|
  |----------(  2 * BLOCK_SIZE2    )------------------->|

  Where we have 2 extra values at each end to avoid boundary conditions in the code
  
*/

__global__ void C6_kernel_shared_f(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT2 * blockIdx.x + threadIdx.x -1)*skip<<1;
  // i = -2*skip 0, 2*skip, ... , len-2*skip, len
  // for each block, we have, e.g. i = -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT2-2*skip, 2*skip*BLOCK_SECT2
  int ish = (threadIdx.x)<<1;
  // ish = 0, 2,... , 2*BLOCK_SIZE2-2, 2*BLOCK_SIZE2
  __shared__ real x_work[BLOCK_SIZE2<<1];
 
  write_wvt_shared_gen(x,bdrs,len,skip,i,ish,2,2,BLOCK_SECT2,x_work);
 
  __syncthreads();

  // in the lifting cycles below, we keep some threads idle so that we only update
  // the intermediate coefficients that we need in the subsequent steps


  //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //if((ish>= )&&(ish<=( ))){  // we use all threads
    //printf("herecyc1 thread%i\n",threadIdx.x);
  lift_cyc_1(x_work,ish,1,FWD);
  
  __syncthreads();

    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(ish<((BLOCK_SIZE2<<1)-2)){ //keep 1 thread idle
    // printf("herecyc2 thread%i\n",threadIdx.x);
    lift_cyc_2(x_work,ish,1,FWD);
  }
  __syncthreads();


  //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(ish<((BLOCK_SIZE2<<1)-2)){ //keep 1 thread idle
    //  printf("herecyc3 thread%i\n",threadIdx.x);
    lift_cyc_3(x_work,ish,1,FWD);
  }
  __syncthreads();


  //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<((BLOCK_SIZE2<<1)-2))){ //keep 2 threads idle
    lift_cyc_4(x_work,ish,1,FWD);
  }
  __syncthreads();

  
  //5th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<((BLOCK_SIZE2<<1)-2))){ // keep 2 threads idle
    lift_cyc_5(x_work,ish,1,FWD);
  }
  __syncthreads();


  // //6th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ - done below
  
  
  // We do last lifting cycle at the same time as writing back to global memory.
  if((ish>=2) && (ish<((BLOCK_SIZE2<<1)-2)) && (i<(int)len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish]*CL6;
    x[i + skip] = x_work[ish+1]*CL7;
  }

  __syncthreads();

}

// same memory diagram as for the forward transform

__global__ void C6_kernel_shared_b(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int i = (BLOCK_SECT2 * blockIdx.x + threadIdx.x -1)*skip<<1;
  // i = -2*skip 0, 2*skip, ... , len-2*skip, len
  // for each block, we have, e.g. i = -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT2-2*skip, 2*skip*BLOCK_SECT2  
  int ish = (threadIdx.x)<<1;
  // ish = 0, 2,... , 2*BLOCK_SIZE2-2, 2*BLOCK_SIZE2
  __shared__ real x_work[BLOCK_SIZE2<<1];
  // printf("\nthreadIdx.x=%i,blockIdx.x=%i,i=%i,ish=%i\n",threadIdx.x,blockIdx.x,i,ish);
  
  // we do the 6th cycle at the same time as reading in the values
  x_work[ish] = get_wvt_shared_gen(x,bdrs,len,skip,i,ish,2,2,BLOCK_SECT2,1)*CL7;
  x_work[ish+1] = get_wvt_shared_gen(x,bdrs,len,skip,i,ish,2,2,BLOCK_SECT2,0)*CL6;
 
  __syncthreads();

  // in the lifting cycles below, we keep some threads idle so that we only update
  // the intermediate coefficients that we need in the subsequent steps
  
  //5th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  if(){ // use all threads
  lift_cyc_5(x_work,ish,1,BWD);
    //  }
  __syncthreads();
  

  //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(ish>=2){ //keep 1 thread idle
    lift_cyc_4(x_work,ish,1,BWD);    
  }
  __syncthreads();


  //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(ish>=2){ //keep 1 thread idle
    //  printf("herecyc3 thread%i\n",threadIdx.x);
    lift_cyc_3(x_work,ish,1,BWD);
  }
  __syncthreads();


    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<((BLOCK_SIZE2<<1)-2))){ //keep 2 threads idle
    // printf("herecyc2 thread%i\n",threadIdx.x);
    lift_cyc_2(x_work,ish,1,BWD);
  }
  __syncthreads();


  //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if((ish>=2)&&(ish<((BLOCK_SIZE2<<1)-2))){ //keep 2 threads idle
    //printf("herecyc1 thread%i\n",threadIdx.x);
    lift_cyc_1(x_work,ish,1,BWD);
  }
  __syncthreads();


  // We write back to global memory.
  if((ish>=2) && (ish<((BLOCK_SIZE2<<1)-2)) && (i<(int)len)){
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

   _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ 
  |        |                                            |        |
  |_ _ _ _ |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ |_ _ _ _ |

  |<---6-->|<---------(2*BLOCK_SECT_ML2)--------------->|<---6-->|
  |<---------------(  2 * BLOCK_SIZE_ML2  )--------------------->|

  indices...
  0, 1, 2, 3, ..., k, k1, ..., k5
  used in 1st level of transform (where k=2*BLOCK_SECT_ML2+6) k1 is k+1 etc

  then indices...
  3, 4, 5, ..., k, k1, ..., k3
  used in 2nd level of transform

  First & last coefficients contain shared memory boundary coefficients* for the transform levels.
  *shared memory boundary coefficients: for the first & last shared memory blocks, they hold periodic boundary coefficient points; for all other blocks, they hold the boundary coefficients of the previous/following memory block.

  */

int C6CUDA_sh_ml2(real* x_d, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=6;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  switch(sense){
  case 1:
    return(fC6CUDAsh_ml2(x_d,len,1,nlevels));
  case 0:
    return(bC6CUDAsh_ml2(x_d,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fC6CUDAsh_ml2(real* x_d, uint len, uint skip, uint nlevels){
  if(skip < (1 << nlevels)){
    cudaError_t cuderr;
    int threadsPerBlock;
    int blocksPerGrid;
 
    uint levels=1; //leave at 1. This is initialisation for level variable!

    uint k1 = 2; //# border coeffs needed for single level kernel
    uint k2 = 8; //# coeffs needed for 2 level kernel

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

      C6_kernel_shared_f<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

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
                  
      C6_kernel_shared_f_ml2<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);
      cudaFree(bdrs);

    }

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in fC6 MLsh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);

    cudaDeviceSynchronize();
    
    return(fC6CUDAsh_ml2(x_d,len,skip<<levels,nlevels));

  }
  return(0);
}


int bC6CUDAsh_ml2(real* x_d, uint len, uint skip){
  if(skip > 0){
    cudaError_t cuderr;
    int threadsPerBlock;
    int blocksPerGrid;
    
    uint levels=1; //leave at 1. This is initialisation for level variable!

    uint k1 = 2; //# border coeffs needed for single level kernel
    uint k2 = 8; //# coeffs needed for 2 level kernel

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

      C6_kernel_shared_b<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);

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
                  
      C6_kernel_shared_b_ml2<<<blocksPerGrid, threadsPerBlock>>>(x_d,len,skip,bdrs,lenb);
      cudaFree(bdrs);

    }

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
        fprintf(stderr, "CUDA error in transform in bC6 MLsh (error code %s)!\n", cudaGetErrorString(cuderr));
        exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();

    // //print stuff...
    // printf("CUDA: len=%u,skip=%u\n",len,skip);
    // printveccu<<<1,1>>>(x_d,len);

    cudaDeviceSynchronize();
    
    return(bC6CUDAsh_ml2(x_d,len,skip>>levels));

  }

  return(0);
}

__global__ void C6_kernel_shared_f_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int a = 8, b = 8;
  int i = (BLOCK_SECT_ML2 * blockIdx.x + threadIdx.x -a/2)*skip << 1;
  // i = -16*skip,..., -2*skip, 0, 2*skip, ... , len-2*skip, len, ... , len + 10*skip
  // for each block, we have, e.g. i = -12*skip, ..., -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT_ML2-2*skip, 2*BLOCK_SECT_ML2*skip, 2*skip*BLOCK_SECT_ML2+2*skip, ..., 2*skip*BLOCK_SECT_ML2+10*skip
  uint ish = (threadIdx.x)<<1;
  uint ishlast = (BLOCK_SIZE_ML2-1)<<1;
  uint li;
  __shared__ real x_work[BLOCK_SIZE_ML2<<1];
  uint skipwork = 1;

  write_wvt_shared_gen(x,bdrs,len,skip,i,ish,a,b,BLOCK_SECT2,x_work);
  __syncthreads();

  for(li = 0; li < 2; li++){
    
    //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-2)&&(ish>=2)) ||
	 ((li==1)&&(ish<=ishlast-4)&&(ish>=4)) ){
      lift_cyc_1(x_work,ish,skipwork,FWD);
    }
    __syncthreads();

    
    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=2)) ||
	((li==1)&&(ish<=ishlast-8)&&(ish>=4)) ){
      lift_cyc_2(x_work,ish,skipwork,FWD);
    }
    __syncthreads();
    
    
    //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=2)) ||
	((li==1)&&(ish<=ishlast-8)&&(ish>=4)) ){
      lift_cyc_3(x_work,ish,skipwork,FWD);
    }
    __syncthreads();
    

    //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-8)&&(ish>=8)) ){
      lift_cyc_4(x_work,ish,skipwork,FWD);
    }
    __syncthreads();
    
    
    //5th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-8)&&(ish>=8)) ){
      lift_cyc_5(x_work,ish,skipwork,FWD);
    }
    __syncthreads();

    
    //6th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-8)&&(ish>=8)) ){
      lift_cyc_6(x_work,ish,skipwork,FWD);
    }
    __syncthreads();
    
    
    if(li==0) ish=ish<<1; skipwork=skipwork<<1;

  }

  
  // Now transform level is done. We copy shared array x_work back into x

  ish = (threadIdx.x)<<1;

  if( (ish>=a) && (ish<2*BLOCK_SECT_ML2+b) && (i<len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skip] = x_work[ish+1];
  }
  
  
}


// not done 2 level backwards yet. Code is currently LA8!


__global__ void C6_kernel_shared_b_ml2(real* x, const uint len, const uint skip, real* bdrs, const uint lenb){
  int a = 8, b = 8;
  int i = (BLOCK_SECT_ML2 * blockIdx.x + threadIdx.x -a/2)*skip;
  // i = -16*skip,..., -2*skip, 0, 2*skip, ... , len-2*skip, len, len +2, ... , len + 16*skip, len + 14*skip
  // for each block, we have, e.g. i = -16*skip, ..., -2*skip, 0, 2*skip, ..., 2*skip*BLOCK_SECT_ML2-2*skip, 2*BLOCK_SECT_ML2*skip, 2*skip*BLOCK_SECT_ML2+2*skip, ..., 2*skip*BLOCK_SECT_ML2+14*skip
  uint skipbl = skip/2; // skip value for second layer. Used in filling shared mem & filling global memory vector.
  uint ish = (threadIdx.x)<<1;
  uint ishlast = min(len/skipbl+a+b,(BLOCK_SIZE_ML2-1)<<1); //size of shared vec x_work
  uint li;

  // ish = 0, 2,... , 2*BLOCK_SECT_ML2-2, 2*BLOCK_SECT_ML2, 2*BLOCK_SECT_ML2+2, 2*BLOCK_SECT_ML2+4, 2*BLOCK_SECT_ML2+6, +8, ... , +38
  __shared__ real x_work[BLOCK_SIZE_ML2<<1];
  uint skipwork = 2;

  write_wvt_shared_gen(x,bdrs,len,skipbl,i,ish,a,b,BLOCK_SECT2,x_work);
  __syncthreads();

  ish=ish<<1; //align ish for 1st level of backward trans

  for(li = 0; li < 2; li++){
   
    //6th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast)&&(ish>=0)) ||
	((li==1)&&(ish<=ishlast-4)&&(ish>=4)) ){
      lift_cyc_6(x_work,ish,skipwork,BWD);
    }
    __syncthreads();


    //5th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast)&&(ish>=0)) ||
	((li==1)&&(ish<=ishlast-4)&&(ish>=4)) ){
      lift_cyc_5(x_work,ish,skipwork,BWD);
    }
    __syncthreads();

    
    //4th cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-4)&&(ish>=6)) ){
      lift_cyc_4(x_work,ish,skipwork,BWD);
    }
    __syncthreads();
    
    
    //3rd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-4)&&(ish>=6)) ){
      lift_cyc_3(x_work,ish,skipwork,BWD);
    }
    __syncthreads();
    

    //2nd cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-6)&&(ish>=6)) ){
      lift_cyc_2(x_work,ish,skipwork,BWD);
    }
    __syncthreads();
           
    
    //1st cycle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if( ((li==0)&&(ish<=ishlast-4)&&(ish>=4)) ||
	((li==1)&&(ish<=ishlast-6)&&(ish>=6)) ){
      lift_cyc_1(x_work,ish,skipwork,BWD);
    }
    __syncthreads();

    
    if(li==0) ish=ish>>1; skipwork=skipwork>>1;

  }

  
  // Now transform level is done. We copy shared array x_work back into x

  ish = (threadIdx.x)<<1;

  if( (ish>=a) && (ish<2*BLOCK_SECT_ML2+b) && (i<len)){
    //  printf("hereputback thread%i\n",threadIdx.x);
    x[i] = x_work[ish];
    x[i + skipbl] = x_work[ish+1];
  }
  
  
  

}


static __device__ void lift_cyc_1(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish+sk] = xsh[ish+sk] + xsh[ish]*CL0;
  else xsh[ish+sk] = xsh[ish+sk] - xsh[ish]*CL0;
  // d1[l] = x[2l+1] + q11*x[2l]
}

static __device__ void lift_cyc_2(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish] = xsh[ish] + xsh[ish+3*sk]*CL1;
  else xsh[ish] = xsh[ish] - xsh[ish+3*sk]*CL1;
  // s1[l] = x[2l] + q21*x[2l+3]
}

static __device__ void lift_cyc_3(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish+sk] = xsh[ish+sk] + xsh[ish]*CL2;
  else xsh[ish+sk] = xsh[ish+sk] - xsh[ish]*CL2;
  // d2[l] = x[2l+1] + q31*x[2l]
}

static __device__ void lift_cyc_4(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish] = xsh[ish] + xsh[ish-sk]*CL3 + xsh[ish+sk]*CL4;
  else xsh[ish] = xsh[ish] - xsh[ish-sk]*CL3 - xsh[ish+sk]*CL4;
  // s2[l] = x[2l] + q41*x[2l-1] + q42*x[2l+1]
}

static __device__ void lift_cyc_5(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD) xsh[ish+sk] = xsh[ish+sk] + xsh[ish]*CL5;
  else xsh[ish+sk] = xsh[ish+sk] - xsh[ish]*CL5;
  // d3[l] = x[2l+1] + s*K^2*x[2l]
}

static __device__ void lift_cyc_6(real* xsh, const int ish, const uint sk, const short int sense){
  if(sense == FWD){
    xsh[ish] = xsh[ish]*CL6;
    xsh[ish+sk] = xsh[ish+sk]*CL7;
  }
  else{
    xsh[ish] = xsh[ish]*CL7;
    xsh[ish+sk] = xsh[ish+sk]*CL6;
  } 
  //s3[l] = (K)*s2[l]
  // d4[l] = (1/K)*d3[l]
}


int C6CUDA_sh_ml2_streams(real* x_h, real* x_d, uint len, short int sense, uint nlevels, cudaStream_t stream){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=6;
  uint ret;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels == 0) return(1); //NB nlevels=0 when calling this function means that check_len_levels will calculate the maximum number of levels - in which case it will return this number
  // however, it the case of an error, it will return 0 - because any strictly positive integer would be valid. & nlevels is unsigned.
  cudaMemcpyAsync(x_d,x_h,len*sizeof(real),HTD,stream);
  switch(sense){
  case 1:
    ret = fC6CUDAsh_ml2(x_d,len,1,nlevels);
    break;
  case 0:
    ret = bC6CUDAsh_ml2(x_d,len,1<<(nlevels-1));
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