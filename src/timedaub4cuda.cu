#include "daub4cuda.cuh"
#include "daub4.h"

// main for timing the daub transform using CUDA, serial & openMP

int main(void){
  real *x_h, *x1_h, *x2_h;
  real /* *x_d, *x1_d, */ *x2_d, *x3_d;
  uint len;
  uint size;
  int res;
  int power, i, reps=30;
  cudaError_t cuderr;
  cudaEvent_t start;
  cudaEvent_t stop;
  float cudatime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double t1, t2;

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

    printf("\nN,Pow,Serialt,OpenMPt,CUDAt,SGFlops,OmpGFlops,CuGFlops,\t=>CUDA speedup\n");

  // for(int i = 0; i<1; i++){
    for(power = 3; power<24; power++){
      len = 1 << power;
      size=len*sizeof(real);
      //   printf("\nLength = %i, i.e. 2^%i\n",len,power);
      printf("%u,%i,",len,power);
      x_h=(real *)malloc(size);
      x1_h=(real *)malloc(size);
      x2_h=(real *)malloc(size);
    
      // we only define x2 on device
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
          
      //   printf("### Serial transform ### \n");
      mptimer(-1);
      for(i=0;i<reps;i++){
	res=Daub4(x_h,len,1,0);
	if(res!=0) printf("\n## Error in transform\n");
	res=Daub4(x_h,len,0,0);
	if(res!=0) printf("\n## Error in transform\n");
      }
      t1=mptimer(1);
      printf("%g,",t1/(double)reps);
      
      
     //   printf("### Omp transform ### \n");
      mptimer(-1);
      for(i=0;i<reps;i++){
	res=lompDaub4(x1_h,len,1,0);
	if(res!=0) printf("\n## Error in transform\n");
	res=lompDaub4(x1_h,len,0,0);
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
	res=Daub4CUDA_sh(x2_d,len,1,0);
	//res=Daub4CUDA_sh_io(x2_d,x3_d,len,1,0);
	//res=Daub4CUDA(x2_d,len,1);
	cudaDeviceSynchronize();
	cuderr = cudaGetLastError();
	if (cuderr != cudaSuccess)
	  {
	    fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
	    exit(EXIT_FAILURE);
	  }
	res=Daub4CUDA_sh_ml2(x2_d,len,0,0);
	//res=Daub4CUDA_sh_io(x3_d,x2_d,len,0,0);
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
      printf("%g,",cudatime/(1000.0*(double)reps));
      /* printf("%g,",2*4*(len-1)/(1e9*t1/(double)reps)); // serial time */
      /* printf("%g,",2*4*(len-1)/(1e9*t2/(double)reps)); // omp time */
      /* printf("%g,",2*4*(len-1)/(1e6*cudatime/(double)reps)); // cuda time */
      printf("\t%g\n",t1*1000/cudatime); //CUDA speedup
      
      cudaMemcpy(x2_h,x2_d,size,DTH);
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy DtH (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}

      //res=cmpvec(x_h,x1_h,len);
      //res=cmpvec(x_h,x2_h,len);
      //      res=cmpvec(x1_h,x2_h,len);

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
      free(x2_h);
    
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
