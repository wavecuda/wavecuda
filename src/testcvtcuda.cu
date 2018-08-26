#include "threshcuda.cuh"
#include "transformcuda.cuh"

int main(void){
  real thresh1, thresh2;
  //uint len = 1024;
  uint pow, p0 = 10;
  uint pmax = 20;
  //uint pmax = p0;
  uint len;
  wst *w_cpu;
  cuwst *w_gpu;
  wst *w_gpu_cpual;
  uint minl, maxl;
  
  short ttype = MODWT_TO;
  //short ttype = DWT;
  short filt = HAAR;
  short filtlen = 2;
  uint levels;
  //short hardness = HARD;
  short hardness = SOFT;

  cudaEvent_t start;
  cudaEvent_t stop;
  float cudatime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double cputime;

  for(pow = p0; pow <= pmax; pow++){
    
    len = (1 << pow);
    levels = check_len_levels(len,0,filtlen);
    
    printf("\n############# New CVT loop #############\n");
    
    printf("\nVector length is %u, which is 2^%u\n",len,pow);

    w_cpu = create_wvtstruct(ttype, filt, filtlen, levels, len);
    w_gpu = create_cuwvtstruct(ttype, filt, filtlen, levels, len);
    
    if(pow<=10){
      read_1darray("noisydopC1024.csv",w_cpu->x,len,(1<<(10-pow)));
    }
    else{
      read_1darray("bignoisydopC.csv",w_cpu->x,len,(1<<(25-pow)));
    }
    //initrandvec(w_cpu->x,len);

    copyvec(w_cpu->x,w_gpu->x_h,len);

    minl = 0;
    maxl = w_cpu->levels-1;
    
    // transform(w_cpu,FWD);
    // thresh1 = univ_thresh(w_cpu,minl,maxl);
    // transform(w_cpu,BWD);
    
    // transform(w_gpu,FWD);
    // w_gpu_cpual = cpu_alias_cuwvtstruct(w_gpu);
    // cudaDeviceSynchronize();
    // thresh2 = univ_thresh(w_gpu_cpual,minl,maxl);
    // transform(w_gpu,BWD);

    // threshold(w_cpu,NULL,thresh1,hardness,minl,maxl);
    // threshold(w_gpu,NULL,thresh2,hardness,minl,maxl,NULL);
    // cudaDeviceSynchronize();
    // update_cuwst_host(w_gpu);
    // cudaDeviceSynchronize();
    
    mptimer(-1);
    printf("\n\nCPU CVT...\n");
    // thresh1=CVT(w_cpu,hardness,0.01,minl,maxl);
    cputime = mptimer(1);
  
    printf("\nCPU threshold is %g\n",thresh1);
    printf("\nCPU time is %g\n",cputime);

    cudaEventRecord(start, NULL);
    printf("\n\nGPU CVT...\n");
    thresh2=CVT(w_gpu,hardness,0.01,minl,maxl);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cudatime, start, stop);
  
    printf("\nGPU threshold is %g\n",thresh2);
    printf("\nGPU time is %g\n",cudatime/1000.);
  
  
    kill_wvtstruct(w_cpu);
    kill_cuwvtstruct(w_gpu);
  
  }

  return(0);
}