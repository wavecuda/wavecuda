#include "c6cuda.cuh"
#include "c6.h"

int main(void){
  real *x_h, *x1_h, *x2_h;
  real /* *x_d, *x1_d,*/ *x2_d, *x3_d;
  uint len;
  uint size;
  int res;
  int power, i, reps=10;
  cudaError_t cuderr;
  cudaEvent_t start;
  cudaEvent_t stop;
  float cudatime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double t1, t2, ops;
  uint levels=0;

  int p0 = 3;
  int p = 25;

  printf("\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("\nC6 Transform - CUDArised\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("%-20s","Compiled:");
  printf("%s, ",__TIME__);
  printf("%s\n",__DATE__);
  printf("%-20s","Written by:");
  printf("%-20s","jw1408@ic.ac.uk\n");
  printf("\n");
  printf("Recall that the lifting version of the algorithm orders coefficients differently from the traditional implementation!\n");
  printf("Flops given are assuming lifting #operations - divide standard implementation flops by 14/24\n");
  for(len=0;len<50;len++) printf("#");

  printf("\nN,Pow,Serialt,OpenMPt,CUDAt,SGFlops,OmpGFlops,CuGFlops,\t=>CUDA speedup\n");

  for(power = p0; power<=p; power++){
    len = 1 << power;
    size=len*sizeof(real);
    // //   printf("\nLength = %i, i.e. 2^%i\n",len,power);
    printf("%u,%i,",len,power);
    x_h=(real *)malloc(size);
    x1_h=(real *)malloc(size);
    x2_h=(real *)malloc(size);
    
    // for(i = 0; i<y_nrows; i++){
    //   cuderr = cudaMalloc((void**)&(y_work[i]),y_ncols * sizeof(double));
    //   if (cuderr != cudaSuccess)
    //   {
    // 	fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
    // 	exit(EXIT_FAILURE);
    //   }
    // }
    

    // we only define x2 on device
    cuderr = cudaMalloc((void **)&x2_d,size);
    cuderr = cudaMalloc((void **)&x3_d,size);
    if (cuderr != cudaSuccess)
      {
    	fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
    	exit(EXIT_FAILURE);
      }

    initrandvec(x_h,len);
    // x_h[0]=0.; x_h[1]=0.; x_h[2]=5.; x_h[3]=4.;
    // x_h[4]=8.; x_h[5]=6.; x_h[6]=7.; x_h[7]=3.;
    //for(i = 0; i<len; i++) x_h[i] = (double)i;
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
    
    //   printf("### Serial transform ### \n");
    mptimer(-1);
    for(i=0;i<reps;i++){
      res=C6(x_h,len,FWD,levels);
      // res=lC6(x_h,len,FWD,levels);
      if(res!=0) printf("\n## Error in transform\n");
      res=C6(x_h,len,BWD,levels);
      // res=lC6(x_h,len,BWD,levels);
      if(res!=0) printf("\n## Error in transform\n");
      //   printf("Time to do forward serial transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
    }
    t1=mptimer(1);
    printf("%g,",t1/(double)reps);

    //   printf("### Omp transform ### \n");
    mptimer(-1);
    for(i=0;i<reps;i++){
      res=lompC6(x1_h,len,FWD,levels);
      if(res!=0) printf("\n## Error in transform\n");
      res=lompC6(x1_h,len,BWD,levels);
      if(res!=0) printf("\n## Error in transform\n");
    }
    t2=mptimer(1);
    //   printf("Time to do forward omp transform: %gs\n",t);
    //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
    printf("%g,",t2/(double)reps);

    //   printf("### CUDA transform ### \n");
    // do CUDA transform
    cudaEventRecord(start, NULL);
    for(i=0;i<reps;i++){
      //res=C6CUDA_sh(x2_d,len,FWD,levels); //lifting with shared memory
      res=C6CUDA_sh_ml2(x2_d,len,FWD,levels); //lifting with shared memory
      cudaDeviceSynchronize();
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
      	{
      	  fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
      	  exit(EXIT_FAILURE);
      	}
      //res=C6CUDA_sh(x2_d,len,BWD,levels); //lifting with shared memory
      res=C6CUDA_sh_ml2(x2_d,len,BWD,levels); //lifting with shared memory
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
    cudaEventElapsedTime(&cudatime, start, stop);
    // printf("Time to do forward CUDA transform: %gs\n",cudatime/1000.0);
    //  printf("Gigaflops: %g\n",2*4*(len-1)/(1e6*cudatime));
    printf("%g,",cudatime/(1000.0*(double)reps));
    
    //print Gflops - serial then CUDA
    ops=(double) len * (double) reps * 2. * 14;// * 2.; // no reverse
    printf("%g,",ops/(1e9*t1)); // serial flops
    printf("%g,",ops/(1e9*t2)); // omp flops
    printf("%g",ops/(1e6*cudatime)); // cuda flops
    printf("\t%g\n",t1*1000/cudatime); //CUDA speedup

    cudaMemcpy(x2_h,x2_d,size,DTH);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in cudaMemcpy DtH (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }

    // printvec(x_h,len);
    
    //res=cmpvec(x_h,x1_h,len);
    //res=cmpvec(x_h,x2_h,len);
    // res=cmpvec(x1_h,x2_h,len);

    // printf("X_CUDA[%i] = %g\n",len-2,x2_h[len-2]);
    // printf("X_CPU[%i] = %g\n",len-2,x_h[len-2]);

    // printf("Res,RCUDA\n");
    // for(uint i=0;i<len;i++) printf("%g,%g\n",x1_h[i],x2_h[i]);

    // printvec(x_h,len);
    // printvec(x2_h,len);

    // axpby(x1_h,1.,x2_h,-1.,x_h,len);
    // printvec(x2_h,len);

    free(x_h);
    free(x1_h);
    free(x2_h);
    
    cudaFree(x2_d);
    cudaFree(x3_d);

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