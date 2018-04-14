#include "daub4cuda.cuh"
#include "daub4.h"

int main(void){
  real *x_h, *x1_h, *x2_h;
  real /* *x_d, *x1_d,*/ *x2_d, *x3_d;
  uint len;
  uint size;
  int res;
  int power, i, reps=1;
  cudaError_t cuderr;
  cudaEvent_t start;
  cudaEvent_t stop;
  float cudatime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double t1, t2, ops;
  uint levels=0;

  int p0 = 3, p = 20;

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
  printf("NB: check numbers for GFLOPS calcs - calculated for forward & inverse transformations?\n");
  printf("\n");
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
      res=lDaub4(x_h,len,1,levels);
      if(res!=0) printf("\n## Error in transform\n");
      //res=lDaub4(x_h,len,0,levels);
      if(res!=0) printf("\n## Error in transform\n");
      //   printf("Time to do forward serial transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
    }
    t1=mptimer(1);
    printf("%g,",t1/(double)reps);

    //   printf("### Omp transform ### \n");
    mptimer(-1);
    for(i=0;i<reps;i++){
      res=lompDaub4(x1_h,len,1,levels);
      if(res!=0) printf("\n## Error in transform\n");
      //res=lompDaub4(x1_h,len,0,levels);
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
      //res=Daub4CUDA(x2_d,len,1,levels); //basic lifting algo
      res=Daub4CUDA_sh(x2_d,len,1,levels); //lifting with shared memory
      //res=Daub4CUDA_sh_ml2(x2_d,len,1,levels); //lifting with shared memory
      //res=Daub4CUDA_sh_ml2(x2_d,len,1,levels); //lifting with shared memory & multi-level
      //res=Daub4CUDA_sh_io(x2_d, x3_d,len,1,levels); //lifting with shared memory & in/output vectors
      //res=Daub4CUDA_sh_ml2_io(x2_d, x3_d,len,1,levels); //lifting with shared memory & in/output vectors
      cudaDeviceSynchronize();
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}
      //res=Daub4CUDA(x2_d,len,0,0); //basic lifting algo
      //res=Daub4CUDA_sh(x2_d,len,0,0); //lifting with shared memory
      //res=Daub4CUDA_sh_ml2(x2_d,len,0,0); //lifting with shared memory & multi-level
      //      res=Daub4CUDA_sh_io(x3_d, x2_d,len,0,levels); //lifting with shared memory & in/output vectors
      //res=Daub4CUDA_sh_ml2_io(x3_d, x2_d,len,0,levels); //lifting with shared memory & in/output vectors
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
    ops=(double) len * (double) reps * 2. * 14. * 2.; // forward & reverse?
    printf("%g,",ops/(1e9*t1)); // serial time
    printf("%g,",ops/(1e9*t2)); // omp time
    printf("%g",ops/(1e6*cudatime)); // cuda time
    printf("\t%g\n",t1*1000/cudatime); //CUDA speedup

    cudaMemcpy(x2_h,x2_d,size,DTH);

    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in cudaMemcpy DtH (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }
   
    
    //res=cmpvec(x_h,x1_h,len);
    res=cmpvec(x_h,x2_h,len);
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