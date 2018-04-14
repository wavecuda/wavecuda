#include "haarcuda.cuh"
#include "haar.h"

int main(void){
  real *x_h, *x1_h, *x2_h;
  real /* *x_d,*/ *x1_d, *x2_d;
  real *xm_h, *xm1_h, *xm2_h;
  real *xm_d, *xm1_d, *xm2_d;
  real *xmh, *xmd;
  uint len, modlen;
  uint size;
  int res;
  int power, i, reps=20, p0=3, p=24;
  cudaError_t cuderr;
  uint levels = 0;
  int  sn = 4;
  int lev;
  int is;
  
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

  // for(int i = 0; i<1; i++){
  for(power = p0; power<p; power++){
    len = 1 << power;
    size=len*sizeof(real);
    printf("\nLength = %i, i.e. 2^%i\n",len,power);
    //printf("%u,%i,",len,power);
    x_h=(real *)malloc(sn*size);
    //x1_h=(real *)malloc(size);
    cudaMallocHost(&x1_h,sn*size); // pinned host memory
    //x2_h=(real *)malloc(size);
    cudaMallocHost(&x2_h,sn*size); // pinned host memory
    // so that we can do streams!

    lev = levels==0 ? power : levels;
    // variable holding actual number of levels performed

    modlen = len*2*lev;

    // xm_h=(real *)malloc(modlen*sizeof(real));
    // xm1_h=(real *)malloc(modlen*sizeof(real));

    // device allocations

    cuderr = cudaMalloc((void **)&x1_d,sn*size);   
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }

    cudaDeviceSynchronize();

    cuderr = cudaMalloc((void **)&x2_d,sn*size);   
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }

    // cuderr = cudaMalloc((void **)&xm1_d,modlen*sizeof(real));   
    // if (cuderr != cudaSuccess)
    //   {
    // 	fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
    // 	exit(EXIT_FAILURE);
    //   }

    cudaDeviceSynchronize();

    initrandvec(x_h,len);
    // x_h[0]=0.; x_h[1]=0.; x_h[2]=5.; x_h[3]=4.;
    // x_h[4]=8.; x_h[5]=6.; x_h[6]=7.; x_h[7]=3.;
    //for(i=0;i<len;i++) x_h[i]=i;

    copyvec(x_h,x1_h,len);
    copyvec(x_h,x2_h,len);

    for(is = 1; is < sn; is++){
      copyvec(x_h,x_h+is*len,len);
      copyvec(x_h,x1_h+is*len,len);
      copyvec(x_h,x2_h+is*len,len);
    }

    // copy x1 from host to device
    cuderr = cudaMemcpy(x1_d,x1_h,sn*size,HTD);
    cuderr = cudaGetLastError();
    if (cuderr != cudaSuccess)
      {
	fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
	exit(EXIT_FAILURE);
      }

    // copy x2 from host to device
    // cuderr = cudaMemcpy(x2_d,x2_h,size,HTD);
    // cuderr = cudaGetLastError();
    // if (cuderr != cudaSuccess)
    //   {
    // 	fprintf(stderr, "CUDA error in cudaMemcpy HtD (error code %s)!\n", cudaGetErrorString(cuderr));
    // 	exit(EXIT_FAILURE);
    //   }

        
    cudaStream_t stream[sn];
    for(i = 0; i < sn; i++)
      cudaStreamCreate(&stream[i]);
  
          
    cudaDeviceSynchronize();

    printf("\n### Serial transform ### \n");

    for(i=0;i<reps;i++){
      for(is = 0; is < sn; is++){
	res=Haar(x_h+len*is,len,FWD,levels);
	if(res!=0) printf("\n## Error in transform\n");
      }
      // res=Haar(x_h,len,0,levels);
      // if(res!=0) printf("\n## Error in transform\n");
	
      // res=HaarMODWT(x_h,&xm_h,len,FWD,levels);
      // if(res!=0) printf("\n## Error in transform\n");

      // res=HaarMODWTto(x_h,xm_h,len,FWD,levels);
      // if(res!=0) printf("\n## Error in transform\n");

      // write_test_modwt(xm_h,len,lev);

      // res=HaarMODWTto(x_h,xm_h,len,BWD,levels);
      // if(res!=0) printf("\n## Error in transform\n");
      
      //   printf("Time to do forward serial transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      
      //   }
 
      // printf("%g,",t1/(double)reps);

      
      printf("### CUDA transform 1 ### \n");

      //   for(i=0;i<reps;i++){
      // 	printf("Forward trans...\n");
      for(is = 0; is < sn; is++){
	res=HaarCUDAML(x1_d+is*len,len,FWD,levels);
      }
      // 	//printf("(not executed!)\n");
      // 	//res=HaarCUDAMODWTv4(x1_h,&xm1_h,len,FWD,levels);
      // res=HaarCUDAMODWTv5(x1_d,xm1_d,len,FWD,levels);
      cudaDeviceSynchronize();
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
      	{
      	  fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
      	  exit(EXIT_FAILURE);
      	}
      
      // write_test_modwt(xm1_h,len,lev);

      cuderr = cudaMemcpy(x1_h,x1_d,sn*size,DTH);
      cudaDeviceSynchronize();

      
      if (cuderr != cudaSuccess)
	{
	  fprintf(stderr, "CUDA error in cudaMemcpy DtH (error code %s)!\n", cudaGetErrorString(cuderr));
	  exit(EXIT_FAILURE);
	}

      
      // res=HaarCUDAMODWTv5(x1_d,xm1_d,len,BWD,levels);
      // cudaDeviceSynchronize();
      // cuderr = cudaGetLastError();
      // if (cuderr != cudaSuccess)
      // 	{
      // 	  fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
      // 	  exit(EXIT_FAILURE);
      // 	}	
      // if(res!=0) printf("\n## Error in transform\n");


      // printf("Backward trans...\n");
      // res=HaarCUDAML(x1_d,len,0,levels);
      // cudaDeviceSynchronize();
      // cuderr = cudaGetLastError();
      // if (cuderr != cudaSuccess)
      //   {
      //     fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
      //     exit(EXIT_FAILURE);
      //   }	
      // if(res!=0) printf("\n## Error in transform\n");
      
      //   }

      //   printf("Time to do forward omp transform: %gs\n",t);
      //    printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
      // printf("%g,",t2/(double)reps);
      

      printf("### CUDA transform 2 ### \n");

      //xm2_h=(real *)malloc(modlen*sizeof(real));
      //  cudaMallocHost(&xm2_h,modlen*sizeof(real)); // pinned host memory
      //xmh=(real *)malloc(sn*modlen*sizeof(real));
      // cudaMallocHost(&xmh,sn*modlen*sizeof(real)); // pinned host memory
      // cuderr = cudaMalloc((void **)&xmd,sn*modlen*sizeof(real));
      // if (cuderr != cudaSuccess)
      //   {
      // 	fprintf(stderr, "CUDA error in Malloc (error code %s)!\n", cudaGetErrorString(cuderr));
      // 	exit(EXIT_FAILURE);
      //   }
      
      
      //  for(i=0;i<reps;i++){

      // if(i%sn==0) cudaDeviceSynchronize();
	
      // printf("Forward trans...\n");
      // res=HaarCUDAML(x1_d,len,1,levels);
      for(is = 0; is < sn; is++){
	res = HaarCUDAMLv2(x2_h+is*len,x2_d+is*len,len,FWD,levels,stream[is]);
      }
      cudaDeviceSynchronize();

      // res=HaarCUDAMODWTv6(x2_h,xmh+(i%sn)*modlen,x2_d,xmd+(i%sn),len,FWD,levels,stream[i%sn]);
	
      // res=HaarCUDAMODWTv6(x2_h,xmh+(i%sn)*modlen,x2_d,xmd+(i%sn),len,BWD,levels,stream[i%sn]);
      

      // res=HaarCUDAMODWTv5(x2_d,xm2_d,len,FWD,levels);
      // cudaDeviceSynchronize();
      cuderr = cudaGetLastError();
      if (cuderr != cudaSuccess)
        {
          fprintf(stderr, "CUDA error in transform (error code %s)!\n", cudaGetErrorString(cuderr));
          exit(EXIT_FAILURE);
        }	
      if(res!=0) printf("\n## Error in transform\n");
      printf("\n## cpp vs cuda 2 ##\n");
      res=cmpvec(x_h,x2_h,sn*len);
      if(res == 0) printf("\nNo errors here!!\n");
      
      printf("\n## cpp vs cuda 1 ##\n");
      res+=cmpvec(x_h,x1_h,sn*len);
      if(res == 0) printf("\nNo errors here either!!\n");

      if(res>0) return(0);
      // if we get errors in the compare vectors, then abandon ship!

	
    }
      
    for (i = 0; i<sn; i++)
      cudaStreamDestroy(stream[i]);


    // printf("Time to do forward CUDA transform: %gs\n",cudatime/1000.0);
    //  printf("Gigaflops: %g\n",2*4*(len-1)/(1e6*cudatime));
    // printf("%g,",cudatime/(1000.0*(double)reps));
            
    //print Gflops - serial then CUDA
    // printf("%g,",2*4*(len-1)/(1e9*t1/(double)reps)); // serial time
    // printf("%g,",2*4*(len-1)/(1e9*t2/(double)reps)); // omp time
    // printf("%g,",2*4*(len-1)/(1e6*cudatime/(double)reps)); // cuda time
    // printf("\t%g\n",t1*1000/cudatime); //CUDA speedup

    // cudaMemcpy(x1_h,x1_d,size,DTH);
    // cuderr = cudaGetLastError();
    // if (cuderr != cudaSuccess)
    //   {
    // 	fprintf(stderr, "CUDA error in cudaMemcpy DtH (error code %s)!\n", cudaGetErrorString(cuderr));
    // 	exit(EXIT_FAILURE);
    //   }

    // cudaMemcpy(xm1_h,xm1_d,modlen*sizeof(real),DTH);
    // if (cuderr != cudaSuccess)
    //   {
    // 	fprintf(stderr, "CUDA error in cudaMemcpy DtH (error code %s)!\n", cudaGetErrorString(cuderr));
    // 	exit(EXIT_FAILURE);
    //   }
    // cudaDeviceSynchronize();

    //printvec(xm2_h,modlen);

    //printvec(xm_h,modlen);
      
    // only copy back xm2 when not already done in the programme!
      
    // cudaMemcpy(xm2_h,xm2_d,modlen*sizeof(real),DTH);
    // cuderr = cudaGetLastError();
    // if (cuderr != cudaSuccess)
    // 	{
    // 	  fprintf(stderr, "CUDA error in cudaMemcpy DtH xm2 (error code %s)!\n", cudaGetErrorString(cuderr));
    // 	  exit(EXIT_FAILURE);
    // 	}

    //res=rejig(x2_h,len,CTI);
    
    // res=cmpvec(xm_h,xm1_h,modlen);
    //   print_modwt_vec_to(xm_h,len,power);
    //res=cmpvec(xm_h,xm2_h,modlen);
    //res=cmpvec(xm_h,xmh,modlen);
    //res=cmpvec(xm_h,xmh+modlen,modlen);
 
    // printf("Res,RCUDA\n");
    // for(uint i=0;i<len;i++) printf("%g,%g\n",x1_h[i],x2_h[i]);

    // printvec(x1_h,len);
    // printvec(x2_h,len);

    // axpby(x1_h,1.,x2_h,-1.,x_h,len);
    // printvec(x2_h,len);

    // printvec(xm_h,modlen);
    // printvec(xm2_h,modlen);
      
    //printf("\n Hello?!\n");

    free(x_h);
    //free(x1_h);
    cudaFreeHost(x1_h);
    //cudaFreeHost(xm1_h);
    // free(xm1_h);
    cudaFreeHost(x2_h);
    //cudaFreeHost(xm2_h);
    // cudaFree(xm1_d);
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
