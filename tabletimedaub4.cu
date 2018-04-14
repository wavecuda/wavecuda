#include "daub4cuda.cuh"
#include "daub4.h"

// main for timing all Daub4 versions in existence, in order to show a pretty graph of relative flops. Just to show how many I've written!


int main(void){
  real *x_h, *x1_h;
  real *x2_d, *x3_d;
  uint len;
  uint size;
  int res;
  int power, i, reps=10; //was 10
  cudaError_t cuderr;
  cudaEvent_t start;
  cudaEvent_t stop;
  float cut1,cut2,cut3,cut4,cut5;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double t1, t2, t3;
  double s_ops, l_ops;
  uint levels = 0;

  printf("\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("\nDaub4 Transform - CUDArised\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("%-20s","Compiled:");
  printf("%s, ",__TIME__);
  printf("%s\n",__DATE__);
  printf("%-20s","Written by:");
  printf("%-20s","jw1408@ic.ac.uk\n");
  printf("\n");
  for(len=0;len<50;len++) printf("#");

    printf("\nN,Pow,Serialt,SerialLiftt,OpenMPt,CUDA1t,CUDAsht,CUDAml2t,CUDAshiot,CUDAml2iot,SGFlops,SLGFlops,OmpGFlops,Cu1GFlops,CUshGFlops,CUml2GFlops,CUshioGFlops,CUml2ioGFlops\n");

    for(power = 3; power<=25; power++){
      len = 1 << power;
      size=len*sizeof(real);
      printf("%u,%i,",len,power);
      x_h=(real *)malloc(size);
      x1_h=(real *)malloc(size);
    
      // we define x2 and x3 on device
      cuderr = cudaMalloc((void **)&x2_d,size);   
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}

      cuderr = cudaMalloc((void **)&x3_d,size);   
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}

      initrandvec(x_h,len);

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
	res=Daub4(x1_h,len,FWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
	res=Daub4(x1_h,len,BWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
	//   printf("Time to do forward serial transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      }
      t1=mptimer(1);
      printf("%g,",t1/(double)reps);
      
      copyvec(x_h,x1_h,len);
      
      //   printf("### Serial lifting transform ### \n");
      mptimer(-1);
      for(i=0;i<reps;i++){
	res=lDaub4(x1_h,len,FWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
	res=lDaub4(x1_h,len,BWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
	//   printf("Time to do forward serial transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      }
      t2=mptimer(1);
      printf("%g,",t2/(double)reps);
      
      copyvec(x_h,x1_h,len);

      //   printf("### Omp transform ### \n");
      mptimer(-1);
      for(i=0;i<reps;i++){
	res=lompDaub4(x1_h,len,FWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
	res=lompDaub4(x1_h,len,BWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
      }
      t3=mptimer(1);
      //   printf("Time to do forward omp transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      printf("%g,",t3/(double)reps);

      
      //   printf("### plain CUDA transform ### \n");
      // do CUDA transform
      cudaEventRecord(start, NULL);
      for(i=0;i<reps;i++){
	res=Daub4CUDA(x2_d,len,FWD,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
	res=Daub4CUDA(x2_d,len,BWD,levels);
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
	res=Daub4CUDA_sh(x2_d,len,FWD,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
	res=Daub4CUDA_sh(x2_d,len,BWD,levels);
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
	res=Daub4CUDA_sh_ml2(x2_d,len,FWD,levels);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
	res=Daub4CUDA_sh_ml2(x2_d,len,BWD,levels);
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


      if(len<(1<<24)){
	//   printf("### IO CUDA transform ### \n");
	// do CUDA transform
	cudaEventRecord(start, NULL);
	for(i=0;i<reps;i++){
	  res=Daub4CUDA_sh_io(x2_d,x3_d,len,FWD,levels);
	  cudaDeviceSynchronize();
	  cuderr = cudaGetLastError();
	  if (cuderr != cudaSuccess)
	    {
	      fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	      exit(EXIT_FAILURE);
	    }
	  res=Daub4CUDA_sh_io(x2_d,x3_d,len,BWD,levels);
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
      }
      else printf("0,");

      // copy x from host to device
      cuderr = cudaMemcpy(x2_d,x_h,size,HTD);
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}

      if(len<(1<<24)){
	//   printf("### IO ML2 CUDA transform ### \n");
	// do CUDA transform
	cudaEventRecord(start, NULL);
	for(i=0;i<reps;i++){
	  res=Daub4CUDA_sh_ml2_io(x2_d,x3_d,len,FWD,levels);
	  cudaDeviceSynchronize();
	  cuderr = cudaGetLastError();
	  if (cuderr != cudaSuccess)
	    {
	      fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	      exit(EXIT_FAILURE);
	    }
	  res=Daub4CUDA_sh_ml2_io(x2_d,x3_d,len,BWD,levels);
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
	cudaEventElapsedTime(&cut5, start, stop);
	// printf("Time to do forward CUDA transform: %gs\n",cudatime/1000.0);
	//  printf("Gigaflops: %g\n",2*4*(len-1)/(1e6*cudatime));
	printf("%g,",cut5/(1000.0*(double)reps));
      }
      else printf("0,");

      s_ops = (double) len * (double) reps * 2. * 14. * 2.;
      l_ops = (double) len * (double) reps * 2. * 9. * 2.;
      
      //print Gflops - serial then CUDA
      printf("%g,",s_ops/(1e9*t1)); // serial time
      printf("%g,",l_ops/(1e9*t2)); // serial lifting time
      printf("%g,",l_ops/(1e9*t3)); // omp time

      printf("%g,",l_ops/(1e6*cut1)); // cuda time
      printf("%g,",l_ops/(1e6*cut2)); // cuda sh time
      printf("%g,",l_ops/(1e6*cut3)); // cuda ML time
      if(len<(1<<24)){
	printf("%g,",l_ops/(1e6*cut4)); // cuda coal sh time
	printf("%g,",l_ops/(1e6*cut5)); // cuda coal ML time
      }
      else printf("0,0,");
      printf("\n");

      free(x_h);
      free(x1_h);
    
      cudaFree(x2_d);
      cudaFree(x3_d);
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaFree (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
    
    }
  return(0);
}
