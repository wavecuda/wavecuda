#include "haarcuda.cuh"
#include "haar.h"

// main for timing all Haar versions in existence, in order to show a pretty graph of relative flops. Just to show how many I've written!

int main(void){
  real *x_h, *x1_h;
  real *x2_d;
  uint len;
  uint size;
  int res;
  int power, i, reps=10; //was 10
  cudaError_t cuderr;
  cudaEvent_t start;
  cudaEvent_t stop;
  float cut1,cut2,cut3,cut4;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double t1, t2, t3;
  uint levels = 0;

  printf("\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("\nHaar Transform - CUDArised\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("%-20s","Compiled:");
  printf("%s, ",__TIME__);
  printf("%s\n",__DATE__);
  printf("%-20s","Written by:");
  printf("%-20s","jw1408@ic.ac.uk\n");
  printf("\n");
  for(len=0;len<50;len++) printf("#");

    printf("\nN,Pow,Serialt,OpenMPt,CUDA1t,CUDAsht,CUDAML2t,CUDAcoalt,SGFlops,OmpGFlops,Cu1GFlops,CUshGFlops,CUml2GFlops,CUcGFlops,\t=>CUDA speedup\n");

  // for(int i = 0; i<1; i++){
    for(power = 3; power<=28; power++){
      len = 1 << power;
      size=len*sizeof(real);
      //   printf("\nLength = %i, i.e. 2^%i\n",len,power);
      printf("%u,%i,",len,power);
      x_h=(real *)malloc(size);
      x1_h=(real *)malloc(size);
    
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

      // copy x from host to device
      cuderr = cudaMemcpy(x2_d,x_h,size,HTD);
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
          

      //   printf("### Serial transform ### \n");
      mptimer(-1);
      for(i=0;i<reps;i++){
	//res=HaarCoalA(x_h,len,0);
	res=Haar(x1_h,len,1,levels);
	if(res!=0) printf("\n## Error in transform\n");
	res=Haar(x1_h,len,0,levels);
	if(res!=0) printf("\n## Error in transform\n");
	//   printf("Time to do forward serial transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      }
      t1=mptimer(1);
      printf("%g,",t1/(double)reps);
      
      copyvec(x_h,x1_h,len);
      
      //   printf("### Omp transform ### \n");
      mptimer(-1);
      for(i=0;i<reps;i++){
	res=Haarmp(x1_h,len,1,levels);
	if(res!=0) printf("\n## Error in transform\n");
	res=Haarmp(x1_h,len,0,levels);
	if(res!=0) printf("\n## Error in transform\n");
      }
      t2=mptimer(1);
      //   printf("Time to do forward omp transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      printf("%g,",t2/(double)reps);
      
      copyvec(x_h,x1_h,len);

      
      // //   printf("### Serial coal transform ### \n");
      // mptimer(-1);
      // for(i=0;i<reps;i++){
      // 	res=HaarCoalA(x1_h,len,1);
      // 	if(res!=0) printf("\n## Error in transform\n");
      // 	res=HaarCoalA(x1_h,len,0);
      // 	if(res!=0) printf("\n## Error in transform\n");
      // 	//   printf("Time to do forward serial transform: %gs\n",t);
      // //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      // }
      // t3=mptimer(1);
      // printf("%g,",t3/(double)reps);
            
      // copyvec(x_h,x1_h,len);

      //   printf("### plain CUDA transform ### \n");
      // do CUDA transform
      cudaEventRecord(start, NULL);
      for(i=0;i<reps;i++){
	res=HaarCUDA(x2_d,len,1,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
	res=HaarCUDA(x2_d,len,0,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
      }      
      cudaEventRecord(stop, NULL);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&cut1, start, stop);
      // printf("Time to do forward CUDA transform: %gs\n",cudatime/1000.0);
      //  printf("Gigaflops: %g\n",2*4*(len-1)/(1e6*cudatime));
      printf("%g,",cut1/(1000.0*(double)reps));

      // copy x from host to device
      cuderr = cudaMemcpy(x2_d,x_h,size,HTD);
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}



      //   printf("### shared mem CUDA transform ### \n");
      // do CUDA transform
      cudaEventRecord(start, NULL);
      for(i=0;i<reps;i++){
	res=HaarCUDAsh(x2_d,len,1,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
	res=HaarCUDAsh(x2_d,len,0,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }	
      }      
      cudaEventRecord(stop, NULL);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&cut2, start, stop);
      // printf("Time to do forward CUDA transform: %gs\n",cudatime/1000.0);
      //  printf("Gigaflops: %g\n",2*4*(len-1)/(1e6*cudatime));
      printf("%g,",cut2/(1000.0*(double)reps));

      // copy x from host to device
      cuderr = cudaMemcpy(x2_d,x_h,size,HTD);
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}


      //   printf("### ML2 CUDA transform ### \n");
      // do CUDA transform
      cudaEventRecord(start, NULL);
      for(i=0;i<reps;i++){
	res=HaarCUDAML(x2_d,len,1,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
	res=HaarCUDAML(x2_d,len,0,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
      }
      cudaEventRecord(stop, NULL);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&cut3, start, stop);
      // printf("Time to do forward CUDA transform: %gs\n",cudatime/1000.0);
      //  printf("Gigaflops: %g\n",2*4*(len-1)/(1e6*cudatime));
      printf("%g,",cut3/(1000.0*(double)reps));

      // copy x from host to device
      cuderr = cudaMemcpy(x2_d,x_h,size,HTD);
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}


      //   printf("### coal CUDA transform ### \n");
      // do CUDA transform
      cudaEventRecord(start, NULL);
      for(i=0;i<reps;i++){
	res=HaarCUDACoalA(x2_d,len,1);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
	res=HaarCUDACoalA(x2_d,len,0);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
      }
      cudaEventRecord(stop, NULL);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&cut4, start, stop);
      // printf("Time to do forward CUDA transform: %gs\n",cudatime/1000.0);
      //  printf("Gigaflops: %g\n",2*4*(len-1)/(1e6*cudatime));
      printf("%g,",cut4/(1000.0*(double)reps));


      //print Gflops - serial then CUDA
      printf("%g,",2*4*(len-1)/(1e9*t1/(double)reps)); // serial time
      printf("%g,",2*4*(len-1)/(1e9*t2/(double)reps)); // omp time
      // printf("%g,",2*4*(len-1)/(1e9*t2/(double)reps)); // serial coal time
      printf("%g,",2*4*(len-1)/(1e6*cut1/(double)reps)); // cuda time
      printf("%g,",2*4*(len-1)/(1e6*cut2/(double)reps)); // cuda sh time
      printf("%g,",2*4*(len-1)/(1e6*cut3/(double)reps)); // cuda ML time
      printf("%g,",2*4*(len-1)/(1e6*cut4/(double)reps)); // cuda coal time

      printf("\n");
      
      free(x_h);
      free(x1_h);
    
      cudaFree(x2_d);
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
