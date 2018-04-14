#include "haar.h"
#include "utils.h"
#include "thresh.h"

int main(void){
  real *x, *x1, *x2, *xm, *xm1, *xm2;
  uint len, modlen;
  int res;
  int power;
  double t,t2,t3;
  uint levels = 0;
  int p0 = 4, p = 5;
  uint i;

  printf("\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("\nHaar Transform in C++ - serial & OpenMP\n");
  for(len=0;len<50;len++) printf("#");
  printf("\n");
  printf("%-20s","Compiled:");
  printf("%s, ",__TIME__);
  printf("%s\n",__DATE__);
  printf("%-20s","Written by:");
  printf("%-20s","jw1408@ic.ac.uk\n");
  printf("\n");
  for(len=0;len<50;len++) printf("#");
  
  // ###################################################

  for(power = p0; power<p; power++){
    len= 1 << power;
    printf("\nLength = %i, i.e. 2^%i\n",len,power);
    x=(real *)malloc(len*sizeof(real));
    x1=(real *)malloc(len*sizeof(real));
    x2=(real *)malloc(len*sizeof(real));
    initrandvec(x,len);

    // x[0]=0.; x[1]=0.; x[2]=5.; x[3]=4.;
    // x[4]=8.; x[5]=6.; x[6]=7.; x[7]=3.;
    
    // for(i=0;i<len;i++) x[i]=i;

    copyvec(x,x1,len);
    copyvec(x,x2,len);
    /* printf("Hello, Haar!\n"); */
    /* printf("Haar transform of x,\n"); */
    /* printf("(%g,%g,%g,%g,%g,%g,%g,%g)",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]); */
    /* printf(" is\n"); */
    // mptimer(-1);
    // printf("\n### Serial transform ### \n");
    // res=Haar(x1,len,1,levels);
    ///    if(res!=0) printf("\n## Error in transform\n");
    // res=Haar(x1,len,0);
    // if(res!=0) printf("\n## Error in transform\n");
    //    t=mptimer(1);
    // printf("Time to do forward & backward serial transform: %gs\n",t);
    // printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
    // mptimer(-1);
    // printf("\n### Coalesced A transform ### \n");
    // res=HaarCoalA(x2,len,1);
    // if(res!=0) printf("\n## Error in transform\n");
    // res=Haarmp(x2,len,0);
    // if(res!=0) printf("\n## Error in transform\n");
    // t=mptimer(1);

    modlen = levels==0?len*2*power : len*2*levels;
    xm=(real *)malloc(modlen*sizeof(real));
    xm1=(real *)malloc(modlen*sizeof(real));
    xm2=(real *)malloc(modlen*sizeof(real));
    

    mptimer(-1);
    //HaarMODWT(x,xm,len,FWD,0);
    //HaarMODWTto(x,xm,len,FWD,0);
    HaarMODWTto(x,xm,len,FWD,0);
    //HaarMODWTto(x,xm,len,BWD,0);
    t=mptimer(1);
    // printvec(xm,modlen);
    
    
    mptimer(-1);
    //HaarMODWTtomp(x,xm2,len,FWD,0);
    HaarMODWTtomp(x1,xm1,len,FWD,0);
    //HaarMODWTtomp(x1,xm1,len,BWD,0);
    t2=mptimer(1);
    //printvec(xm2,len*2*3);

    mptimer(-1);
    HaarMODWT(x,xm2,len,FWD,0);
    t3=mptimer(1);
    //printvec(xm3,len*2*3);
    
    // printf("Time to do forward MODWT\n");
    // printf("serial: %g\n",t);
    // printf("omp1: %g\n",t2);
    //printf("omp2: %g\n",t3);
        
    // res=cmpvec(x,x2,len);
    // res=cmpvec(x1,x2,len);
    
    print_modwt_vec_to(xm,len,check_len_levels(len,levels,2));
    // print_modwt_vec_to(xm1,len,check_len_levels(len,levels,2));
    
    print_modwt_vec_po(xm2,len,check_len_levels(len,levels,2));

    // printf("Time to do forward & backward coal transform: %gs\n",t);
    // printf("Gigaflops: %g\n",2*4*(len-1)/(1e9*t));
    /* printf("(%g,%g,%g,%g,%g,%g,%g,%g)\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]); */
    /* res=Haar(x,len,0); */
    /* if(res!=0) printf("\n## Error in transform\n"); */
  
    // res=cmpvec(xm,xm2,modlen);
    // res=cmpvec(xm,xm3,modlen);
    
    /* printf("Inverse Haar transform of x is\n"); */
    /* printf("(%g,%g,%g,%g,%g,%g,%g,%g)\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]); */
    
    free(x);
    free(x1);
    free(x2);
    free(xm);
    free(xm1);
  }
  return(0);
}
