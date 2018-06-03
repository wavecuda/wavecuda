#include "threshcuda.cuh"

#define BLOCK_SIZE 256 // block size for thresholding kernel
//#define BLOCK_SIZE 16 // small size for debugging!

__global__ void thresh_dwt_cuda_kernel_hard(real* in, real* out, const real thresh, const uint len, const uint minlevel, const uint maxlevel){
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < len){
    if(is_in_d_level_limits(i,len,minlevel,maxlevel)){
      // then we are safe to threshold
      out[i] = thresh_coef_cuda_hard(in[i],thresh);
    }else{
      // we are not thresholding the coefficient
      // but if we are copying values, then we need to update
      out[i]=in[i];
    }
  }
}

__global__ void thresh_dwt_cuda_kernel_soft(real* in, real* out, const real thresh, const uint len, const uint minlevel, const uint maxlevel){
  uint i = blockDim.x * blockIdx.x + threadIdx.x;

  uint tid = threadIdx.x;
  __shared__ real in_shared[BLOCK_SIZE];

  if(i < len){
    if(is_in_d_level_limits(i,len,minlevel,maxlevel)){
      in_shared[tid] = in[i];
      // then we are safe to threshold
      out[i] = thresh_coef_cuda_soft(in_shared[tid],thresh);
    }else{
      // we are not thresholding the coefficient
      // but if we are copying values, then we need to update
      out[i]=in[i];
    }
  }
}

__global__ void thresh_modwt_cuda_kernel(real* in, real* out, const real thresh, const uint len, short hardness, const short modwttype, const uint minlevel, const uint maxlevel, const uint levels){
  uint i_smo, i_det;
  uint tid = threadIdx.x;
  extern __shared__ real in_shared[]; // this is shared memory
  // it is dynamically allocated outside the kernel
  // as we want to use it for soft thresholding
  // but not hard thresholding
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  switch(modwttype){
  case MODWT_PO:
    // smoothing coefficients are at even indices up to 2*len for each level
    i_smo = i << 1;
    // detail coefficients are at odd indices up to 2*len for each level
    i_det = i_smo +1;
    break;
  case MODWT_TO:
    // smoothing coefficients are at all indices from 0 to len-1 for each level
    i_smo = i;
    // detail coefficients are at all indices from len to 2*len-1 for each level
    i_det = i_smo + len;
    break;
  }
  uint l, il;
  
  if(i < len){
    for(l=0; l < levels; l++){
      // we loop through the levels
      il = 2*l*len; // base of loop counter   
      __syncthreads();
      if( (l >= minlevel) && (l<=maxlevel)){
	// if between min & max level then we have thresholding to do!
	switch(hardness){
	case HARD:
	  out[il + i_det] = thresh_coef_cuda_hard(in[il+i_det],thresh);
	  out[il + i_smo] = in[il+i_smo];
	  break;
	case SOFT:
	  in_shared[tid] = in[il+i_det];
	  out[il + i_det] = thresh_coef_cuda_soft(in_shared[tid],thresh);
	  out[il + i_smo] = in[il+i_smo];
	}
      }
      else{ // if l not between min & max level
	out[il + i_det] = in[il + i_det];
	out[il + i_smo] = in[il + i_smo];
      }// else
    } // for l loop
  }
}

__device__ real thresh_coef_cuda_hard(const real coef_in, const real thresh){
  if(fabs(coef_in) < thresh) return(0.);
  else return(coef_in);
}

__device__ real thresh_coef_cuda_soft(const real coef_in, const real thresh){
  if(fabs(coef_in) < thresh) return(0.);
  else return(((coef_in < 0) ? -1 : 1)*(fabs(coef_in) - thresh));
}

__device__ double atomicAdd(double* address, double val) {
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


void threshold(cuwst* win, cuwst* wout, real thresh, short hardness, uint minlevel, uint maxlevel, cudaStream_t stream){
  uint len = win->len;
  if(check_len_levels(len,win->levels,minlevel,maxlevel,win->filtlen) > 0){
    // our len, flen, levels, minlevel & maxlevel are compatible!
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (len + BLOCK_SIZE - 1)/BLOCK_SIZE;
    // set up CUDA variables
    // we want len threads in total
    uint shmemsize = hardness==SOFT? BLOCK_SIZE*sizeof(real) : 0;
    // for soft thresholding, we want shared memory
    // for hard thresholding, we do not
    if(win->ttype == DWT){
      // if we are thresholding in place
      switch(hardness){
      case HARD:
	if(wout==NULL){
	  thresh_dwt_cuda_kernel_hard<<<blocksPerGrid, threadsPerBlock,0,stream>>>(win->x_d,win->x_d,thresh,win->len,minlevel,maxlevel);
	}
	else{
	  thresh_dwt_cuda_kernel_hard<<<blocksPerGrid, threadsPerBlock,0,stream>>>(win->x_d,wout->x_d,thresh,win->len,minlevel,maxlevel);
	  wout->transformed = 1;
	}
	break;
      case SOFT:
	if(wout==NULL){
	  // if we are thresholding in place
	  thresh_dwt_cuda_kernel_soft<<<blocksPerGrid, threadsPerBlock,0,stream>>>(win->x_d,win->x_d,thresh,win->len,minlevel,maxlevel);
	}
	else{
	  thresh_dwt_cuda_kernel_soft<<<blocksPerGrid, threadsPerBlock,0,stream>>>(win->x_d,wout->x_d,thresh,win->len,minlevel,maxlevel);
	  wout->transformed = 1;
	  // we update the ouput vector to say it is transformed
	}
	break;
      } // switch hardness
    } // if DWT
    if((win->ttype == MODWT_TO) || (win->ttype == MODWT_PO)){
      if(wout==NULL){
	// if we are thresholding in place
	thresh_modwt_cuda_kernel<<<blocksPerGrid, threadsPerBlock,shmemsize,stream>>>(win->xmod_d,win->xmod_d,thresh,win->len,hardness,win->ttype,minlevel,maxlevel,win->levels);
      }
      else{
	thresh_modwt_cuda_kernel<<<blocksPerGrid, threadsPerBlock,shmemsize,stream>>>(win->xmod_d,wout->xmod_d,thresh,win->len,hardness,win->ttype,minlevel,maxlevel,win->levels);
	wout->transformed = 1;
      }
    }// if MODWT
  }
}

real interp_mse(cuwst* wn, cuwst* wye, cuwst* wyo, mtype *m_d, cudaStream_t stream){
  // GPU version of interp mse function
  // wrapper to GPU kernel

  // calculates interpolation error
  // comparing noisy wn to
  // smoothed ye & yo
  // i.e. interpolate ye & compare to odd values in w
  // & yo with even w
  
  uint len = wn->len;

  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = ((len>>1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  mtype m_h; // the mse
  
  cudaMemsetAsync(m_d,0,sizeof(mtype),stream);
  
  interp_mse_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(wn->x_d,wye->x_d,wyo->x_d,len,m_d);

  //cudaStreamSynchronize(stream);
  
  cudaMemcpyAsync(&m_h,m_d,sizeof(mtype),DTH,stream);
  
  cudaStreamSynchronize(stream);

  // we only want to return our m_h once the kernel & memcpy have finished!
  
  return((real)(0.5*m_h));
  
}

__global__ void interp_mse_kernel(real* xn, real* ye, real* yo, uint len, mtype* m_global){
  // shared memory for the yie & yio values
  // with some atomic add for the m value at the end
  // currently global memory reads for interpolation - no need to space out
  // to prevent bank conflicts

  // __shared__ mtype m_shared;
  __shared__ real y_interp[BLOCK_SIZE<<1];
  // this will contain each respective yie, yio BLOCK_SIZE times
  
  uint j = blockDim.x * blockIdx.x + threadIdx.x;
  // j goes from 0 to len/2 - 1, it is our counter in ye & yo
  uint i = j << 1;
  // i goes from 0 to len-1, it is our counter in xn
  uint ish_e = threadIdx.x << 1;
  uint ish_o = ish_e + 1;
  // ish_e & _o are indices of the shared memory y_interp
  uint skip = 2;
  
  // if(threadIdx.x==0){
  //   m_shared = 0;}
  // __syncthreads();  

  if(i<len){
    if(i==0){
      y_interp[ish_e] = yo[0];
      y_interp[ish_o] = 0.5*(ye[j] + ye[j+1]);
    }
    if(i==len-2){
      y_interp[ish_e] = 0.5*(yo[j] + yo[j-1]);
      y_interp[ish_o] = ye[len/2-1];
    }
    if((i>0) && (i<len-2)){
      y_interp[ish_e] = 0.5*(yo[j] + yo[j-1]);
      y_interp[ish_o] = 0.5*(ye[j] + ye[j+1]);
    }
    
    // y_interp[ish_e] = y_interp[ish_e]-xn[i];
    // y_interp[ish_o] = y_interp[ish_o]-xn[i+1];
    // y_interp[ish_e] = y_interp[ish_e] * y_interp[ish_e] + y_interp[ish_o] * y_interp[ish_o];
    // these operations calculate the mse component for each i
    // & put the answer in y_interp[ish_e]
    
    y_interp[ish_e] = (y_interp[ish_e]-xn[i])*(y_interp[ish_e]-xn[i]) + (y_interp[ish_o]-xn[i+1])*(y_interp[ish_o]-xn[i+1]);
  }
  __syncthreads();
  // now we reduce a bit...
  // this bit is reliant on the blocksize being a power of two.
  // while(skip <= BLOCK_SIZE){
  //   if( ((i+skip)<len) && ((ish_e+skip)<(BLOCK_SIZE<<1)) ){
  //     if((ish_e % (skip<<1))==0)
  // 	y_interp[ish_e] = y_interp[ish_e] + y_interp[ish_e + skip];
  //   }
  //   skip = skip <<1;
  //   __syncthreads();
  // }
  sum_reduce_shmem(y_interp,skip,len,BLOCK_SIZE<<1,i,ish_e);

  // by the end of this loop, we should have the mse in y_interp[0]
  // of each block


  //atomicAdd(&m_shared,(mtype)y_interp[ish_e]);
    //atomicAdd(&m_shared,(mtype)((y_interp[ish_e]-xn[i])*(y_interp[ish_e]-xn[i]) + (y_interp[ish_o]-xn[i+1])*(y_interp[ish_o]-xn[i+1])));
  //}// if i<len
//__syncthreads();
  // atomic add to m_shared

  // for our 'old' GPUs, (compute 3.0/3.5, only float atomicAdd is hardware supported)
  // if this is not accurate enough, we can always implement a double version
  // as outlined in the programming guide
  // which would give a reduction in performance but an increase in accuracy
    
  if(threadIdx.x==0)
    //atomicAdd(m_global,m_shared);
    atomicAdd(m_global,(mtype)y_interp[0]);
  // atomic add of m_shared to m_global if tid=0
}

real CVT_old(cuwst *w, short hardness, real tol, uint minlevel, uint maxlevel){
  // old version of CVT function using lots of memory transfers between host & device
  // they are done asynchronously so as to minimise time, but apparently that was not sufficient!

  // w->x_h is original (noisy) vector

  real R = 0.61803399, C = 1. - R;

  real ta=0., tb, tc;
  cuwst *y1e, *y1o, *y2e, *y2o;
  cuwst *we, *wo, *wcpy;
  wst *y1e_cpu, *y1o_cpu, *y2e_cpu, *y2o_cpu;
  wst *w_cpu, *we_cpu, *wo_cpu, *wcpy_cpu;
  real t0, t1, t2, t3;
  real m1, m2;
  // ?a is lb, ?c is ub
  // ?0, ?1, ?2, ?3 denote ordered values that we are keeping track of
  // t? is threshold, m? is associated mse
  int res;
  uint len = w->len;
  uint lenh = len >> 1; // len/2
  uint maxlevel1;
  uint levels = w->levels;
  uint iter = 0;
  uint i, sn = 2;

  cudaStream_t stream[sn];
  for(i = 0; i < sn; i++)
    cudaStreamCreate(&stream[i]);
  
  if(check_len_levels(w->len,w->levels,minlevel,maxlevel,w->filtlen) == 0){
    // error with levels
    return(0);
  }

  if(w->levels<=1){
    printf("\nNeed to be transforming at least one level for CVT!");
    return(0);
  }

  // allocate memory for auxiliary wavelet structs
  // these will be rewritten in the main loop
  y1e=create_cuwvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  y1o=create_cuwvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  y2e=create_cuwvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  y2o=create_cuwvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);

  // malloc w: we/o will hold un-thresholded wavelet wavelet coefficients of odd/even separated vectors
  we=create_cuwvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  wo=create_cuwvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);

  copyvecskip(w->x_h,2,len,we->x_h,1); // copy the even indices into we
  copyvecskip(w->x_h+1,2,len,wo->x_h,1); // copy the odd indices into wo
  
  wcpy = dup_cuwvtstruct(w); // we keep a copy of (untransformed) w to avoid extra transforms
  
  // now we create a load of cpu alias versions of the cuwst objects
  // the wavelet transforms will be done to the cuwst objects
  // but they will
  // a) copy x_h -> x_d
  // b) transform x_d in place (DWT) or to xmod_d (MODWT)
  // c) copy x_d -> x_h (DWT) or xmod_d -> xmod_h (MODWT)
  // these aliases allow us to work with arrays allocated on the CPU
  // seeing as they are always copied during transformations
  w_cpu = cpu_alias_cuwvtstruct(w);
  wcpy_cpu = cpu_alias_cuwvtstruct(wcpy);
  we_cpu = cpu_alias_cuwvtstruct(we);
  wo_cpu = cpu_alias_cuwvtstruct(wo);
  y1e_cpu = cpu_alias_cuwvtstruct(y1e);
  y1o_cpu = cpu_alias_cuwvtstruct(y1o);
  y2e_cpu = cpu_alias_cuwvtstruct(y2e);
  y2o_cpu = cpu_alias_cuwvtstruct(y2o);
  
  res=transform(w,FWD);
  cudaDeviceSynchronize();
  w_cpu->transformed = w->transformed;
  tc = univ_thresh(w_cpu,minlevel,maxlevel);
  
  tb=tc/2.; // arbitrary threshold between ta & tc


  // makes points t2 & t1 st t0 to t1 is the smaller segment
  t0=ta;
  t2=tb;
  t1=tb - C*(tb - ta);
  t3=tc;
  
  maxlevel1 = maxlevel -1;

  // (forward) transform w vectors
  // threshold using t1 & t2 
  // (inverse) transform y to x_hat (where x_hat is stored in y!)
  // all in a funny order to maximise concurrency
  
  res=transform(we,FWD,stream[0]);
  res=transform(wo,FWD,stream[1]);
  cudaStreamSynchronize(stream[0]);
  threshold(we_cpu,y1e_cpu,t1,hardness, minlevel, maxlevel1);
  y1e->transformed = y1e_cpu->transformed;
  res=transform(y1e,BWD,stream[0]);
  cudaStreamSynchronize(stream[1]);
  threshold(wo_cpu,y1o_cpu,t1,hardness, minlevel, maxlevel1);
  y1o->transformed = y1o_cpu->transformed;
  res=transform(y1o,BWD,stream[1]);
  // no extra synchronising needed for following step
  // as we already have we transformed
  threshold(we_cpu,y2e_cpu,t2,hardness, minlevel, maxlevel1);
  y2e->transformed = y2e_cpu->transformed;
  res=transform(y2e,BWD,stream[0]);
  // no extra synchronising needed for following step
  // as we already have wo transformed
  threshold(wo_cpu,y2o_cpu,t2,hardness, minlevel, maxlevel1);
  y2o->transformed = y2o_cpu->transformed;
  res=transform(y2o,BWD,stream[1]);  

  cudaStreamSynchronize(stream[0]);
  m1 = interp_mse(wcpy_cpu, y1e_cpu, y1o_cpu);
  
  cudaStreamSynchronize(stream[1]);
  m2 = interp_mse(wcpy_cpu, y2e_cpu, y2o_cpu);

  // minimise MSE by golden search...

  while(fabs(t3-t0) > tol*(t1+t2)){
    if(iter>50){
      printf("\nWe probably aren't converging. Exiting...\n");
      break;
    }
    printf("\nt0 = %g, t1 = %g, t2 = %g, t3 = %g",t0,t1,t2,t3);
    printf("\nm1 = %g, m2 = %g",m1,m2);
    if(m2 < m1){
      // m2 is new curr min, rearrange held points
      t0 = t1; t1=t2; t2=R*t2 + C*t3;
      m1=m2;
      threshold(we_cpu,y2e_cpu,t2,hardness,minlevel,maxlevel1);
      y2e->transformed = y2e_cpu->transformed;
      res=transform(y2e,BWD,stream[0]);
      threshold(wo_cpu,y2o_cpu,t2,hardness,minlevel,maxlevel1);
      y2o->transformed = y2o_cpu->transformed;
      res=transform(y2o,BWD,stream[1]);
      cudaDeviceSynchronize();
      m2=interp_mse(wcpy_cpu,y2e_cpu,y2o_cpu);
    }
    else{
      t3=t2; t2=t1; t1=R*t1 + C*t0;
      m2=m1;
      threshold(we_cpu,y1e_cpu,t1,hardness,minlevel,maxlevel1);
      y1e->transformed = y1e_cpu->transformed;
      res=transform(y1e,BWD,stream[0]);
      threshold(wo_cpu,y1o_cpu,t1,hardness,minlevel,maxlevel1);
      y1o->transformed = y1o_cpu->transformed;     
      res=transform(y1o,BWD,stream[1]);
      cudaDeviceSynchronize();
      m1=interp_mse(wcpy_cpu,y1e_cpu,y1o_cpu);
    }
    iter++;
  }
  
  tc = m1<m2? t1 : t2;
  // tc now contains the chosen threshold
  tc = tc/(sqrt(1. - log(2.)/log((double)len)));
  // scale the threshold to the original, full, data

  threshold(w_cpu,NULL,tc,hardness,minlevel,maxlevel);
  res = transform(w,BWD);
  
  kill_alias_wvtstruct(y1e_cpu);
  kill_alias_wvtstruct(y1o_cpu);
  kill_alias_wvtstruct(y2e_cpu);
  kill_alias_wvtstruct(y2o_cpu);
  kill_alias_wvtstruct(we_cpu);
  kill_alias_wvtstruct(wo_cpu);
  kill_alias_wvtstruct(wcpy_cpu);

  cudaDeviceSynchronize();
  
  kill_cuwvtstruct(y1e);
  kill_cuwvtstruct(y1o);
  kill_cuwvtstruct(y2e);
  kill_cuwvtstruct(y2o);
  kill_cuwvtstruct(we);
  kill_cuwvtstruct(wo);
  kill_cuwvtstruct(wcpy);

  for (i = 0; i<sn; i++)
    cudaStreamDestroy(stream[i]);
  
  
  printf("\n");
  return(tc);

}


real CVT(cuwst *w, short hardness, real tol, uint minlevel, uint maxlevel){
  // new version using far fewer memory transfers between host & device

  // w->x_h is original (noisy) vector

  real R = 0.61803399, C = 1. - R;

  real ta=0., tb, tc;
  cuwst *y1e, *y1o, *y2e, *y2o;
  cuwst *we, *wo, *wcpy;
  real t0, t1, t2, t3;
  real m1, m2;
  mtype *m1_d, *m2_d; // on the gpu
  // ?a is lb, ?c is ub
  // ?0, ?1, ?2, ?3 denote ordered values that we are keeping track of
  // t? is threshold, m? is associated mse
  int res;
  uint len = w->len;
  uint lenh = len >> 1; // len/2
  uint maxlevel1;
  uint iter = 0;
  uint i, sn = 2;
  real *tmp;

  short filt_nohost = -(w->filt); // the versions of wavelet transforms not using host memory are encoded with negative integers; those using host memory & transfers are positive integers

  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  // improve shared memory for use with doubles

  cudaStream_t stream[sn];
  for(i = 0; i < sn; i++)
    cudaStreamCreate(&stream[i]);
  
  cudaMalloc((void **)&m1_d,sizeof(mtype));
  cudaMalloc((void **)&m2_d,sizeof(mtype));

  if(check_len_levels(w->len,w->levels,minlevel,maxlevel,w->filtlen) == 0){
    // error with levels
    return(0);
  }

  if(w->levels<=1){
    printf("\nNeed to be transforming at least one level for CVT!");
    return(0);
  }

  // allocate memory for auxiliary wavelet structs
  // these will be rewritten in the main loop
  y1e=create_cuwvtstruct(w->ttype,filt_nohost,w->filtlen,w->levels-1,lenh);
  y1o=create_cuwvtstruct(w->ttype,filt_nohost,w->filtlen,w->levels-1,lenh);
  y2e=create_cuwvtstruct(w->ttype,filt_nohost,w->filtlen,w->levels-1,lenh);
  y2o=create_cuwvtstruct(w->ttype,filt_nohost,w->filtlen,w->levels-1,lenh);

  // malloc w: we/o will hold un-thresholded wavelet wavelet coefficients of odd/even separated vectors
  we=create_cuwvtstruct(w->ttype,filt_nohost,w->filtlen,w->levels-1,lenh);
  wo=create_cuwvtstruct(w->ttype,filt_nohost,w->filtlen,w->levels-1,lenh);

  tmp = (real *)malloc(lenh * sizeof(real));
  copyvecskip(w->x_h,2,len,tmp,1); // copy the even indices into tmp
  cudaMemcpy(we->x_d,tmp,lenh*sizeof(real),HTD); // copy tmp into we
  copyvecskip(w->x_h+1,2,len,tmp,1); // copy the odd indices into tmp
  cudaMemcpy(wo->x_d,tmp,lenh*sizeof(real),HTD); // copy tmp into wo
  free(tmp);
  
  w->filt = filt_nohost; // change w to allow for manual copying

  update_cuwst_device(w); // copy w->x_h to w->x_d
  cudaDeviceSynchronize();
  wcpy = dup_cuwvtstruct(w,1,0); // we keep a copy of (untransformed) w to avoid extra transforms
  // with options yes to memcpy & no to host allocation
  
  
  // now we create a cpu alias version of one of the the cuwst objects
  // these aliases allow us to work with arrays allocated on the CPU
  // seeing as they are always copied during transformations with memcpy
  
  res=transform(w,FWD);
  //update_cuwst_host(w);
  cudaDeviceSynchronize();
  tc = univ_thresh_approx(w,minlevel,maxlevel,stream[0]);
  
  tb=tc/2.; // arbitrary threshold between ta & tc

  // makes points t2 & t1 st t0 to t1 is the smaller segment
  t0=ta;
  t2=tb;
  t1=tb - C*(tb - ta);
  t3=tc;
  
  maxlevel1 = maxlevel -1;

  // (forward) transform w vectors
  // threshold using t1 & t2 
  // (inverse) transform y to x_hat (where x_hat is stored in y!)
  // all in a funny order to maximise concurrency
  
  res=transform(we,FWD,stream[0]);
  res=transform(wo,FWD,stream[1]);
  cudaDeviceSynchronize();
  threshold(we,y1e,t1,hardness, minlevel, maxlevel1,stream[0]);
  res=transform(y1e,BWD,stream[0]);
  threshold(wo,y1o,t1,hardness, minlevel, maxlevel1,stream[0]);
  res=transform(y1o,BWD,stream[0]);
  // no extra synchronising needed for following step
  // as we already have we transformed
  threshold(we,y2e,t2,hardness, minlevel, maxlevel1,stream[1]);
  res=transform(y2e,BWD,stream[1]);
  // no extra synchronising needed for following step
  // as we already have wo transformed
  threshold(wo,y2o,t2,hardness, minlevel, maxlevel1,stream[1]);
  res=transform(y2o,BWD,stream[1]);  

  m1 = interp_mse(wcpy, y1e, y1o,m1_d,stream[0]);
  
  m2 = interp_mse(wcpy, y2e, y2o,m2_d,stream[1]);

  cudaDeviceSynchronize();
  
  // minimise MSE by golden search...

  while(fabs(t3-t0) > tol*(t1+t2)){
    if(iter>50){
      printf("\nWe probably aren't converging. Exiting...\n");
      break;
    }
    // printf("\nt0 = %g, t1 = %g, t2 = %g, t3 = %g",t0,t1,t2,t3);
    // printf("\nm1 = %g, m2 = %g",m1,m2);
    if(m2 < m1){
      // m2 is new curr min, rearrange held points
      t0 = t1; t1=t2; t2=R*t2 + C*t3;
      m1=m2;
      threshold(we,y2e,t2,hardness,minlevel,maxlevel1,stream[0]);
      res=transform(y2e,BWD,stream[0]);
      threshold(wo,y2o,t2,hardness,minlevel,maxlevel1,stream[1]);
      res=transform(y2o,BWD,stream[1]);
      cudaDeviceSynchronize();
      m2=interp_mse(wcpy,y2e,y2o,m2_d,NULL);
      cudaDeviceSynchronize();
    }
    else{
      t3=t2; t2=t1; t1=R*t1 + C*t0;
      m2=m1;
      threshold(we,y1e,t1,hardness,minlevel,maxlevel1,stream[0]);
      res=transform(y1e,BWD,stream[0]);
      threshold(wo,y1o,t1,hardness,minlevel,maxlevel1,stream[1]);
      res=transform(y1o,BWD,stream[1]);
      cudaDeviceSynchronize();
      m1=interp_mse(wcpy,y1e,y1o,m1_d,NULL);
      cudaDeviceSynchronize();
    }
    iter++;
  }
  
  tc = m1<m2? t1 : t2;
  // tc now contains the chosen threshold
  tc = tc/(sqrt(1. - log(2.)/log((double)len)));
  // scale the threshold to the original, full, data
  
  threshold(w,NULL,tc,hardness,minlevel,maxlevel,NULL);
  // threshold w on the GPU
  w->filt = filt_nohost;
  res = transform(w,BWD);
  update_cuwst_host(w);
  // this copies the GPU x_d values
  // into the CPU x_h
  w->filt = -filt_nohost; // restore value of w->filt
  
  kill_cuwvtstruct(y1e);
  kill_cuwvtstruct(y1o);
  kill_cuwvtstruct(y2e);
  kill_cuwvtstruct(y2o);
  kill_cuwvtstruct(we);
  kill_cuwvtstruct(wo);
  kill_cuwvtstruct(wcpy,0);
  // kill cuwst, but no host alloc so don't try to free that

  cudaFree(m1_d);
  cudaFree(m2_d);

  for (i = 0; i<sn; i++)
    cudaStreamDestroy(stream[i]);
  
  // printf("\n");
  cudaDeviceSynchronize();
  return(tc);

}


__device__ short is_in_d_level_limits(uint i, uint len, uint minlevel, uint maxlevel){
  // short function to ascertain whether an index i of a DWT vector is
  // a detail coefficient inside the limits (for thresholding)
  uint im = i % (2<<maxlevel); // i mod (2<<maxlevel) should be a non zero multiple of min_skip
  uint min_skip = 1<<minlevel; // the skip value at minlevel
  return((im/min_skip)>0);
}

__global__ void sum_n_sqdev_dwt_details(real* x, const uint len, real *sum, const real *mean, const short ttype, const uint minlevel, const uint maxlevel, uint n_det, const short sqdev){
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  uint tid = threadIdx.x;
  __shared__ real x_work[BLOCK_SIZE];
  uint skip = 1;
  
  if(i < len){
    if(is_in_d_level_limits(i,len,minlevel,maxlevel)){
      x_work[tid] = x[i];
      if(sqdev)
	x_work[tid] = (x_work[tid] - (*mean))*(x_work[tid] - (*mean));
      // if we are working out the square deviances, then do that
      // else, we are just working out the sum of the detail coeffs
    }
    else{
      x_work[i] = 0;
    }
  }
  __syncthreads();
  
  sum_reduce_shmem(x_work,skip,len,BLOCK_SIZE,i,tid);
  // puts the sum of the details into x_work[0]
  
  __syncthreads();

  if(tid==0){
    if(sqdev)
      atomicAdd(sum,x_work[0]/(real)(n_det-1));
    else
      atomicAdd(sum,x_work[0]/(real)n_det);
    // atomic add of block's sum/n to global sum if tid=0
  }
  
}


__global__ void sum_n_sqdev_modwt_details(real* x, const uint len, real *sum, const real *mean, const short modwttype, const uint minlevel, const uint maxlevel, const uint n_det, const short sqdev){
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  uint i_det;
  uint tid = threadIdx.x;
  uint l, il;
  uint skip = 1;
  __shared__ real x_work[BLOCK_SIZE];
  switch(modwttype){
  case MODWT_PO:
    // detail coefficients are at odd indices up to 2*len for each level
    i_det = (i << 1) +1;
    break;
  case MODWT_TO:
    // detail coefficients are at all indices from len to 2*len-1 for each level
    i_det = i + len;
    break;
  }
  
  if(i < len){
    for(l= minlevel; l <= maxlevel; l++){
      // we loop through the levels
      il = 2*l*len; // base of loop counter   
      __syncthreads();
      if(l == minlevel)
	x_work[tid] = 0;
      // initialise shared memory array
      
	if(sqdev){ // we are working out sq deviation from mean
	  x_work[tid] += (x[il+i_det] - (*mean))*(x[il+i_det] - (*mean));
	}
	else{ // we are just working out sum
	  x_work[tid] += x[il+i_det];
	}
    } // for l
  }// if i
  __syncthreads();
  
  sum_reduce_shmem(x_work,skip,len,BLOCK_SIZE,i,tid);
  // puts the sum of the details into x_work[0]
  
  __syncthreads();

  if(tid==0){
    if(sqdev)
      atomicAdd(sum,x_work[0]/(real)(n_det-1));
    else
      atomicAdd(sum,x_work[0]/(real)n_det);
    // atomic add of block's sum to global sum if tid=0
  }

}

__device__ void sum_reduce_shmem(real* xsh, uint skip, uint len, uint sh_size, uint i, uint ish){
  while(skip <= (sh_size>>1)){
    if( ((i+skip)<len) && ((ish+skip)<sh_size) ){
      if((ish % (skip<<1))==0)
	xsh[ish] = xsh[ish] + xsh[ish + skip];
    }
    skip = skip <<1;
    __syncthreads();
  }
}

real univ_thresh_approx(cuwst *w, uint minlevel, uint maxlevel, cudaStream_t stream){
  real *meanvar_d; // mean & var on the device
  real var_h; // var on the host
  int threadsPerBlock;
  int blocksPerGrid;
  uint n_det; // number of detail coeffs in thresholding levels
  cudaMalloc((void **)&meanvar_d,2*sizeof(real));
  cudaMemsetAsync(meanvar_d,0,2*sizeof(real),stream);

  threadsPerBlock = BLOCK_SIZE;
  blocksPerGrid = (w->len + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // set up CUDA variables
  
  n_det  = ndetail_thresh(w,minlevel, maxlevel);

  switch(w->ttype){
  case DWT:
    sum_n_sqdev_dwt_details<<<blocksPerGrid,threadsPerBlock,0,stream>>>(w->x_d,w->len,meanvar_d,0,w->ttype,minlevel,maxlevel,n_det,0);
    // this puts the mean of the thresholding details into meanvar_d[0]

    // now we have the mean, we want to find the variance

    sum_n_sqdev_dwt_details<<<blocksPerGrid,threadsPerBlock,0,stream>>>(w->x_d,w->len,meanvar_d+1,meanvar_d,w->ttype,minlevel,maxlevel,n_det,1);
    // this puts the var of the thresholding details into meanvar_d[1]
    break;
  case MODWT_TO:
  case MODWT_PO:
    sum_n_sqdev_modwt_details<<<blocksPerGrid,threadsPerBlock,0,stream>>>(w->xmod_d,w->len,meanvar_d,0,w->ttype,minlevel,maxlevel,n_det,0);
    // this puts the mean of the thresholding details into meanvar_d[0]

    // now we have the mean, we want to find the variance
    
    sum_n_sqdev_modwt_details<<<blocksPerGrid,threadsPerBlock,0,stream>>>(w->xmod_d,w->len,meanvar_d+1,meanvar_d,w->ttype,minlevel,maxlevel,n_det,1);
    // this puts the var of the thresholding details into meanvar_d[1]
    break;
  }
  
  cudaMemcpyAsync(&var_h,&meanvar_d[1],sizeof(real),DTH,stream);
  
  cudaStreamSynchronize(stream);
  
  cudaFree(meanvar_d);

  return(sqrt(var_h * 2.*log((double)(w->len))));
}