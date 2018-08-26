#include "c6.h"
#include "utils.h"
#include "thresh.h"


int main(void){
  // real x[8]={1,1,0,0,0,0,0,0};
  // real y[8]={1,1,0,0,0,0,0,0};
  real *x, *x1;
  real *y;
  int res;
  int pow = 15;
  uint len=1 << pow;

  x=(real *)malloc(len*sizeof(real));
  y=(real *)malloc(len*sizeof(real));
  x1=(real *)malloc(len*sizeof(real));
  
  //for(int i=0;i<len;i++) x[i]=i;
  initrandvec(x,len);
  // x[0]=0.; x[1]=0.; x[2]=5.; x[3]=4.;
  // x[4]=8.; x[5]=6.; x[6]=7.; x[7]=3.;

  copyvec(x,y,len);
  copyvec(x,x1,len);
  

  printf("\nC6 transform - tradition pyramid DWT");
  res=C6(x,len,FWD,0);
  // printvec(x,len);

  // printf("\nC6 transform inverse - tradition pyramid DWT");
  // res=C6(x,len,BWD,0);
  // printvec(x,len);
  
  printf("\nC6 transform - brand spanking new lifting version");
  res=lC6(y,len,FWD,0);
  // printf("\nshifted version to match...");
  // shift_wvt_vector(y,len,1,-1,-1);
  // shift_wvt_vector(y,len,2,-1,-1);
  // shift_wvt_vector(y,len,4,-1,-1);

  // printvec(y,len);

  // printf("\nLA8 transform - lifting inverse test");
  // res=lC6(y,len,BWD,0);
  // printvec(y,len);


  res = cmpvec(x,y,len);
  

  // shift_wvt_vector(x1,len,2,-1,2);
  // printvec(x1,len);

  // shift_wvt_vector(x1,len,2,1,-2);
  // printvec(x1,len);
  // shifting vector appears to be working


  // printf("\nLA8 transform - lifting inverse filter");
  // res=lLA8(y,len,BWD,0);
  // printvec(y,len);

  // free(x);
  // free(y);
  // free(x1);
  
  printf("\n");
  return(0);
}
