#include "wavutilscuda.cuh"

__global__ void get_bdrs_sh_k(real* x, const uint len, const uint skip, real* bdrs, const uint lenb, const uint k, const uint block_sect){
  // we form a vector of coefficients at the borders, to be used for both the forwards & backwards transforms
  // so we collect k values for the start & the end of each block
  uint j;
  int i = (BS_BD * blockIdx.x + threadIdx.x) << 1;
  int i2k = ((BS_BD * blockIdx.x + threadIdx.x) <<1) *k; //this has values of multiples of 2k. NB not i*(2k) !! it is i*k.
  //NB for future reference, max int is 32767 - so this will need to be long int or something for k>16
  if(i < (lenb/k)){
    if(i2k==0){
      for(j = 0; j < k; j++){
	bdrs[i2k + j] = x[len - (k-j)*skip];
      }
    }
    else{
      for(j = 0; j < k; j++){
	bdrs[i2k + j] = x[skip*(block_sect*i - (k-j))];
      }
    }
    
    if(i2k==lenb-(k<<1)){
      for(j = 0; j < k; j++){
	bdrs[i2k+j+k] = x[j*skip];
      }
    }
    else{
      for(j = 0; j < k; j++){
	bdrs[i2k+j+k] = x[skip*(block_sect*(i+2) + j)];
      }
    }
  }
    
  //__syncthreads();
}

__device__ double get_wvt_shared_gen(real* x, real* bdrs, const uint len, const uint skip, const int i, const int ish, const uint a, const uint b, const uint bsect, const short isskip){
  // a is # boundary coefficients at the lower end of the shared vector
  // b is # boundary coeffs at the upper end of shared vector

  // threadIdx.x goes from 0, ..., BLOCK_SIZE-1 (which varies according to implementation)
  //             = 0, ,,,, BLOCK_SECT + a + b -1
  // bsect is local BLOCK_SECT value

  uint m = a > b ? a: b; //set m to be the max of a & b
  // as our bdrs vector has 2*m entries for each block.

  // copying x into shared array x_work
  // NB - conditioning on threadIdx.x, not ish!
  if(threadIdx.x < a/2){ //ish == 0, 2, ..., 2a-2
    if(i<(int)len){
      // we are filling the shared block with lower boundary points
      if(isskip) return(bdrs[ish + blockIdx.x*m*2]);
      else return(bdrs[1 + ish + blockIdx.x*m*2]);
    }
  }
  
  if((threadIdx.x >= a/2) && (threadIdx.x < bsect+a/2)){
    // needs to be conditional on i and BLOCK_SECT
    if(i < (int)len){
      // we fill the central block of shared memory (no boundary coeffs)
      if(isskip) return(x[i]);
      else return(x[i + skip]);
    }
    else if((i>=(int)len)&&(i<(int)len+b*skip)){
      // this happens when len < BLOCK_SECT
      // we have to deal with upper boundary points
      if(isskip) return(bdrs[(i-len)/skip + m + blockIdx.x*m*2]);
      else return(bdrs[(i-len)/skip + m+1 + blockIdx.x*m*2]);
    }
  }
  
  if(threadIdx.x >= bsect+a/2){
    // we pick up the end borders
    if((i>=len)&&(i<len+b*skip)){
      if(isskip) return(bdrs[ish - (bsect<<1) - a + m + blockIdx.x*2*m]);
      //here, ish - (BLOCK_SECT_ML2<<1)  - a = (i-len)/skip
      //but the former should be faster to calculate!
      else return(bdrs[ish - (bsect<<1) - a + m + 1 + blockIdx.x*2*m]);
    }
    if(i<len){
      if(isskip) return(bdrs[(i/skip - (bsect<<1)*(blockIdx.x+1)) + m + blockIdx.x*2*m]);
      else return(bdrs[(i/skip - (bsect<<1)*(blockIdx.x+1)) +m+1+blockIdx.x*2*m]);
    }
  }

  return(9999);
}



__device__ void write_wvt_shared_gen(real* x, real* bdrs, const uint len, const uint skip, const int i, const int ish, const uint a, const uint b, const uint bsect, real * x_sh){
  // a is # boundary coefficients at the lower end of the shared vector
  // b is # boundary coeffs at the upper end of shared vector

  // threadIdx.x goes from 0, ..., BLOCK_SIZE-1 (which varies according to implementation)
  //             = 0, ,,,, BLOCK_SECT + a + b -1
  // bsect is local BLOCK_SECT value

  uint m = a > b ? a: b; //set m to be the max of a & b
  // as our bdrs vector has 2*m entries for each block.

  // copying x into shared array x_work
  // NB - conditioning on threadIdx.x, not ish!
  if(threadIdx.x < a/2){ //ish == 0, 2, ..., 2a-2
    if(i<(int)len){
      // we are filling the shared block with lower boundary points
      x_sh[ish] = bdrs[ish + blockIdx.x*m*2];
      x_sh[ish+1] = bdrs[1 + ish + blockIdx.x*m*2];
    }
  }
  
  if((threadIdx.x >= a/2) && (threadIdx.x < bsect+a/2)){
    // needs to be conditional on i and BLOCK_SECT
    if(i < (int)len){
      // we fill the central block of shared memory (no boundary coeffs)
      x_sh[ish] = x[i];
      x_sh[ish+1] = x[i+skip];
    }
    else if((i>=(int)len)&&(i<(int)len+b*skip)){
      // this happens when len < BLOCK_SECT
      // we have to deal with upper boundary points
      x_sh[ish] = bdrs[(i-len)/skip + m + blockIdx.x*m*2];
      x_sh[ish+1] = bdrs[(i-len)/skip + m+1 + blockIdx.x*m*2];
    }
  }
  
  if(threadIdx.x >= bsect+a/2){
    // we pick up the end borders
    if((i>=len)&&(i<len+b*skip)){
      x_sh[ish] = bdrs[ish - (bsect<<1) - a + m + blockIdx.x*2*m];
      //here, ish - (BLOCK_SECT_ML2<<1)  - a = (i-len)/skip
      //but the former should be faster to calculate!
      x_sh[ish+1] = bdrs[ish - (bsect<<1) - a + m + 1 + blockIdx.x*2*m];
    }
    if(i<len){
      x_sh[ish] = bdrs[(i/skip - (bsect<<1)*(blockIdx.x+1)) + m + blockIdx.x*2*m];
      x_sh[ish+1] = bdrs[(i/skip - (bsect<<1)*(blockIdx.x+1)) +m+1+blockIdx.x*2*m];
      
    }
  }
}

cuwst* create_cuwvtstruct(short ttype, short filt, uint filtlen, uint levels, uint len){
  return(create_cuwvtstruct(ttype, filt, filtlen, levels, len,1));
}

cuwst* create_cuwvtstruct(short ttype, short filt, uint filtlen, uint levels, uint len, short hostalloc){
  cudaError_t cuderr;
  cuwst *w;
  cuderr = cudaMallocHost(&w,sizeof(cuwst));
  double *x_d, *xmod_d;
  double *x_h, *xmod_h;
  levels = check_len_levels(len,levels,filtlen);
  if(levels == 0) return(NULL);
  // this is an error trap for incompatible level/len/filtlen
  if(hostalloc){
    cuderr = cudaMallocHost(&x_h,len*sizeof(real)); // pinned host memory
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in MallocHost (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
  }
  cuderr = cudaMalloc((void **)&x_d,len*sizeof(real)); 
  if (cuderr != cudaSuccess)
    {
      fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
      exit(EXIT_FAILURE);
    }
  if((ttype == MODWT_TO) || (ttype == MODWT_PO)){
    // if the transform type is MODWT, then we need to
    // allocate the modwt vector
    if(hostalloc){
      cuderr = cudaMallocHost(&xmod_h,len*2*levels*sizeof(real)); // pinned host memory
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in MallocHost (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
    }
    cuderr = cudaMalloc((void **)&xmod_d,len*2*levels*sizeof(real));  
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
  }

  w->x_d = x_d;
  w->x_h = x_h;
  w->ttype = ttype;
  w->filt = filt;
  w->filtlen = filtlen;
  w->transformed = 0;
  w->levels = levels;
  w->len = len;
  w->xmod_d = xmod_d;
  w->xmod_h = xmod_h;
  return(w);
}

void kill_cuwvtstruct(cuwst *w){
  kill_cuwvtstruct(w,1);
}

void kill_cuwvtstruct(cuwst *w, short hostallocated){
  if(hostallocated){
    cudaFreeHost(w->x_h);
  }
  cudaFree(w->x_d);
  if((w->ttype == MODWT_TO) || (w->ttype == MODWT_PO)){
    if(hostallocated){
      cudaFreeHost(w->xmod_h);
    }
    cudaFree(w->xmod_d);
  }
  cudaFreeHost(w);
}

cuwst* dup_cuwvtstruct(cuwst *w1, short memcpy, short hostalloc){
  cuwst *w2;
  w2 = create_cuwvtstruct(w1->ttype,w1->filt,w1->filtlen,w1->levels,w1->len,hostalloc);
  // allocates cuwvtstruct & sets non-pointer components
  // uses negative of filter code as that is the code for no host memory in the transform

  if(memcpy){
    // then we copy the array elements across too
    // (we might avoid doing this when we don't have to)
    if(hostalloc){
      cudaMemcpy(w2->x_h,w1->x_h,w1->len*sizeof(real),HTH);
      //copyvec(w1->x_h,w2->x_h,w1->len);
    }
    cudaMemcpy(w2->x_d,w1->x_d,w1->len*sizeof(real),DTD);
    if((w1->ttype == MODWT_TO) || (w1->ttype == MODWT_PO)){
      if(hostalloc){
	cudaMemcpy(w2->xmod_h,w1->xmod_h,(w1->len)*2*(w1->levels)*sizeof(real),HTH);
	//copyvec(w1->xmod_h,w2->xmod_h,(w1->len)*2*(w1->levels));
      }
      cudaMemcpy(w2->xmod_d,w1->xmod_d,(w1->len)*2*(w1->levels)*sizeof(real),DTD);
    }
  }
  return(w2);
}

cuwst* dup_cuwvtstruct(cuwst *w1){
  // this behaviour duplicates with memcpy of pointer components and host alloc
  return(dup_cuwvtstruct(w1,1,1));
}

wst* cpu_alias_cuwvtstruct(cuwst* w_gpu){
  wst* w_cpu = (wst *)malloc(sizeof(wst));
  w_cpu->x = w_gpu->x_h;
  w_cpu->ttype = w_gpu->ttype;
  w_cpu->filt = w_gpu->filt;
  w_cpu->filtlen = w_gpu->filtlen;
  w_cpu->transformed = w_gpu->transformed;
  w_cpu->levels = w_gpu->levels;
  w_cpu->len = w_gpu->len;
  w_cpu->xmod = w_gpu->xmod_h;
  return(w_cpu);
}

void kill_alias_wvtstruct(wst *w){
  // the other components will be freed in original GPU structure
  free(w);
}

void print_cuwst_info(cuwst *w){
  printf("\n--------------------------");
  printf("\nGPU wavelet structure");
  printf("\n-------------------------------");
  printf("\nFilter: ");
  switch(w->filt){
  case HAAR: 
  case HAARMP:
    printf("Haar"); break;
  case DAUB4:
  case DAUB4MP:
    printf("Daub4"); break;
  default:
    printf("Unknown filter"); break;
  }
  printf("\nTransform type: ");
  switch(w->ttype){
  case DWT: printf("DWT"); break;
  case MODWT_TO: printf("MODWT, time ordered"); break;
  case MODWT_PO: printf("MODWT, packet ordered"); break;
  default:
    printf("Unknown transform type"); break;
  }
  printf("\nLevels: %u",w->levels);
  printf("\nLength: %u",w->len);
  printf("\n-------------------------------");
  printf("\n");
}

void update_cuwst_host(cuwst *w){
  // function for doing appropriate cudamemcpy ops
  // written for debugging

  // if modwt, then we update either xmod or x,
  // depending on whether w is transformed

  switch(w->ttype){
  case DWT:
    cudaMemcpy(w->x_h,w->x_d,w->len*sizeof(real),DTH);
    break;
  case MODWT_TO:
  case MODWT_PO:
    if(w->transformed){
      cudaMemcpy(w->xmod_h,w->xmod_d,w->len*w->levels*2*sizeof(real),DTH);
    }
    else{
      // we have probably already done a BWD transform
      // so we only need x_d
      cudaMemcpy(w->x_h,w->x_d,w->len*sizeof(real),DTH);
    }
    break;
  }
}

void update_cuwst_device(cuwst *w){
  // function for doing appropriate cudamemcpy ops
  // written for debugging

  // if modwt, then we update either xmod or x,
  // depending on whether w is transformed

  switch(w->ttype){
  case DWT:
    cudaMemcpy(w->x_d,w->x_h,w->len*sizeof(real),HTD);
    break;
  case MODWT_TO:
  case MODWT_PO:
    if(w->transformed){
      cudaMemcpy(w->xmod_d,w->xmod_h,w->len*w->levels*2*sizeof(real),HTD);
    }
    else{
      // we have probably already done a BWD transform
      // so we only need x_d
      cudaMemcpy(w->x_d,w->x_h,w->len*sizeof(real),HTD);
    }
    break;
  }
}

uint ndetail_thresh(cuwst* w, uint minlevel, uint maxlevel){
  //wrapper to function in wavutils
  return(ndetail_thresh(w->ttype,w->len,minlevel,maxlevel));
}
