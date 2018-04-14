#include "haarcuda.cuh"
#include "haar.h"

// main for timing the haar transform using CUDA, serial & openMP
// NB with empty Haar kernel, we get (artificial) 30 GFLOPS - (unattainable) upper bound

int main(void){
  real *x_h, *x1_h, *x2_h;
  real /* *x_d, *x1_d, */ *x2_d;
  real *xm_h, *xm1_h, *xm2_h;
  real /* *xm_d, *xm1_d,*/ *xm2_d;
  uint len, modlen;
  uint size, nlevels;
  int res;
  int power, i;
  cudaError_t cuderr;
  cudaEvent_t start;
  cudaEvent_t stop;
  float cudatime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double t1, t2;

  // ###########################################
  // A few variables to control the timing runs
  //
  int p0 = 3, p = 20; // first & last size in loop
  int reps = 10; // repetitions
  int modwt = 1; // running DWT or MODWT?
  int inverse = 1; // inverse transform too?
  uint levels = 0; // levels of transform
  //
  // ###########################################


  printf("\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("\nHaar Transform - CUDArised\n");
  printf("This run: ");
  if(modwt) printf("MODWT, ");
  else printf("DWT, ");
  if(!inverse) printf("no ");
  else printf("with ");
  printf("inverse\n");
  printf("from len = 2^%d to len = 2^%d\n",p0,p);
  printf("with %d repetitions\n",reps);
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("%-20s","Compiled:");
  printf("%s, ",__TIME__);
  printf("%s\n",__DATE__);
  printf("%-20s","Written by:");
  printf("%-20s","jw1408@ic.ac.uk\n");
  printf("\n");
  for(len=0;len<50;len++) printf("#");

    printf("\nN,Pow,Serialt,OpenMPt,CUDAt,SGFlops,OmpGFlops,CuGFlops,\t=>CUDA speedup\n");
    
    for(power = p0; power<=p; power++){
      len = 1 << power;
      size=len*sizeof(real);
      //   printf("\nLength = %i, i.e. 2^%i\n",len,power);
      printf("%u,%i,",len,power);
      x_h=(real *)malloc(size);
      x1_h=(real *)malloc(size);
      cudaMallocHost(&x2_h,size); // pinned host memory
      //x2_h=(real *)malloc(size);
      
      nlevels = check_len_levels(len,levels,2);
      // we want this for the purpose of time/flops calculation
      if(modwt) modlen = len*2*nlevels;
      
      // we only define x2 on device
      cuderr = cudaMalloc((void **)&x2_d,size);   
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}

      initrandvec(x_h,len);
      // x_h[0]=0.; x_h[1]=0.; x_h[2]=5.; x_h[3]=4.;
      // x_h[4]=8.; x_h[5]=6.; x_h[6]=7.; x_h[7]=3.;
      copyvec(x_h,x1_h,len);
      copyvec(x_h,x2_h,len);

      // copy x2 from host to device
      cuderr = cudaMemcpy(x2_d,x2_h,size,HTD);
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}

      if(modwt){
	xm_h=(real *)malloc(modlen*sizeof(real));
	xm1_h=(real *)malloc(modlen*sizeof(real));
	cudaMallocHost(&xm2_h,modlen*sizeof(real)); // pinned host memory
	// for the CUDA transform
	cudaMalloc((void **)&xm2_d,modlen*sizeof(real));
      }
          
      //   printf("### Serial transform ### \n");
      mptimer(-1);
      for(i=0;i<reps;i++){
	if(!modwt){
	  res=Haar(x_h,len,1,levels);
	  if(res!=0) printf("\n## Error in transform\n");
	  if(inverse){
	    res=Haar(x_h,len,0,levels);
	    if(res!=0) printf("\n## Error in transform\n");
	  } //inverse
	} // !modwt
	else{//modwt
	  res=HaarMODWTto(x_h,xm_h,len,FWD,levels);
	  //res=HaarMODWTomp2(x_h,xm_h,len,FWD,levels);
	  if(res!=0) printf("\n## Error in transform\n");
	  //if(i<(reps-1)) free(xm_h); // we allocate outside the loop
	  if(inverse){
	    res=HaarMODWTto(x_h,xm_h,len,BWD,levels);
	    if(res!=0) printf("\n## Error in transform\n");
	  } //inverse
	} //modwt
	//   printf("Time to do forward serial transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      }
      t1=mptimer(1);
      printf("%g,",t1/(double)reps);
      

      
      //   printf("### Omp transform ### \n");
      mptimer(-1);
      for(i=0;i<reps;i++){
	if(!modwt){
	  res=Haarmp(x1_h,len,1,levels);
	  if(res!=0) printf("\n## Error in transform\n");
	  if(inverse){
	    res=Haarmp(x1_h,len,0,levels);
	    if(res!=0) printf("\n## Error in transform\n");
	  } //inverse
	} // !modwt
	else{//modwt
	  res=HaarMODWTtomp(x_h,xm1_h,len,FWD,levels);	  
	  if(res!=0) printf("\n## Error in transform\n");
	  //if(i<(reps-1)) free(xm1_h); // we allocate outside the loop
	  if(inverse){
	    res=HaarMODWTtomp(x_h,xm1_h,len,BWD,levels);	  
	    if(res!=0) printf("\n## Error in transform\n");
	  } //inverse
	} //modwt
      }
      t2=mptimer(1);
      //   printf("Time to do forward omp transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      printf("%g,",t2/(double)reps);
      

      //   printf("### CUDA transform ### \n");
      // do CUDA transform
      cudaEventRecord(start, NULL);
      for(i=0;i<reps;i++){
	if(!modwt){
	  //res=HaarCUDACoalA(x2_d,len,0);
	  res=HaarCUDAML(x2_d,len,1,levels);
	  cudaDeviceSynchronize();
	  cuderr = cudaGetLastError();
	  if (cuderr != cudaSuccess)
	    {
	      fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	      exit(EXIT_FAILURE);
	    }
	  if(inverse){
	    res=HaarCUDAML(x2_d,len,0,levels);
	    cudaDeviceSynchronize();
	    cuderr = cudaGetLastError();
	    if (cuderr != cudaSuccess)
	      {
		fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
		exit(EXIT_FAILURE);
	      }
	  } //inverse
	} // !modwt
	else{//modwt
	  res=HaarCUDAMODWTv5(x2_d,xm2_d,len,FWD,levels);
	  cudaDeviceSynchronize();
	  cuderr = cudaGetLastError();
	  if (cuderr != cudaSuccess)
	    {
	      fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	      exit(EXIT_FAILURE);
	    }	
	  if(res!=0) printf("\n## Error in transform\n");
	  //if(i<(reps-1)) cudaFree(xm2_d); // we allocate outside the loop
	  if(inverse){
	    res=HaarCUDAMODWTv5(x2_d,xm2_d,len,BWD,levels);
	    cudaDeviceSynchronize();
	    cuderr = cudaGetLastError();
	    if (cuderr != cudaSuccess)
	      {
		fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
		exit(EXIT_FAILURE);
	      }
	  } //inverse
	} //modwt
	
      }
      
      cudaEventRecord(stop, NULL);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&cudatime, start, stop);
      // printf("Time to do forward CUDA transform: %gs\n",cudatime/1000.0);
      //  printf("Gigaflops: %g\n",2*4*(len-1)/(1e6*cudatime));
      printf("%g,",cudatime/(1000.0*(double)reps));
            
      //print Gflops - serial then CUDA
      if(!modwt){
	printf("%g,",(1+inverse)*2*4*(len-1)/(1e9*t1/(double)reps)); // serial time
	printf("%g,",(1+inverse)*2*4*(len-1)/(1e9*t2/(double)reps)); // omp time
	printf("%g,",(1+inverse)*2*4*(len-1)/(1e6*cudatime/(double)reps)); // cuda time
	printf("\t%g\n",t1*1000/cudatime); //CUDA speedup
      }
      else{//modwt
	printf("%g,",(1+inverse)*4*len*nlevels/(1e9*t1/(double)reps)); // serial time
	printf("%g,",(1+inverse)*4*len*nlevels/(1e9*t2/(double)reps)); // omp time
	printf("%g,",(1+inverse)*4*len*nlevels/(1e6*cudatime/(double)reps)); // cuda time
	printf("\t%g\n",t1*1000/cudatime); //CUDA speedup
      }
      
      if(!modwt){
	cudaMemcpy(x2_h,x2_d,size,DTH);
      }
      else{//modwt
	cudaMemcpy(xm2_h,xm2_d,modlen*sizeof(real),DTH);
      }
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy DtH (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}

      if(!modwt){
	res=cmpvec(x_h,x1_h,len);
	res=cmpvec(x_h,x2_h,len);
	res=cmpvec(x1_h,x2_h,len);
      }
      else{//modwt
	res=cmpvec(xm_h,xm1_h,modlen);
	res=cmpvec(xm_h,xm2_h,modlen);
	res=cmpvec(xm1_h,xm2_h,modlen);
      }

      // printf("Res,RCUDA\n");
      // for(uint i=0;i<len;i++) printf("%g,%g\n",x1_h[i],x2_h[i]);

      // printvec(x1_h,len);
      // printvec(x2_h,len);

      // axpby(x1_h,1.,x2_h,-1.,x_h,len);
      // printvec(x2_h,len);

      // printvec(x_h,len);
      // printvec(x2_h,len);
      
      //printf("\n Hello?!\n");

      free(x_h);
      free(x1_h);
      cudaFreeHost(x2_h);
      
      cudaFree(x2_d);
      if(modwt){
	free(xm_h);
	free(xm1_h);
	cudaFreeHost(xm2_h);
	cudaFree(xm2_d);	
      }
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaFree (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      // printf("\n\n");    
    }
    //  }
  return(0);
}
