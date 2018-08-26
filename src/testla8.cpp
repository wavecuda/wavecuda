#include "la8.h"
#include "utils.h"
#include "thresh.h"

int main(void){
  // real x[8]={1,1,0,0,0,0,0,0};
  // real y[8]={1,1,0,0,0,0,0,0};
  real *x, *x1;
  real *y;
  int res;
  int pow = 20;
  uint len=1 << pow;

  x=(real *)malloc(len*sizeof(real));
  y=(real *)malloc(len*sizeof(real));
  x1=(real *)malloc(len*sizeof(real));
  
  for(int i=0;i<len;i++) x[i]=i;
  copyvec(x,y,len);
  copyvec(x,x1,len);
  

  printf("\nLA8 transform - tradition pyramid DWT");
  res=LA8(x,len,FWD,2);
  //printvec(x,len);

  // printf("\nLA8 transform inverse - tradition pyramid DWT");
  // res=LA8(x,len,BWD,2);
  // printvec(x,len);
  
  printf("\nLA8 transform - brand spanking new lifting version");
  res=lompLA8(y,len,FWD,2);
  printf("\nshifted version to match...");
  shift_wvt_vector(y,len,1,-3,0);
  shift_wvt_vector(y,len,2,-3,0);
  //printvec(y,len);

  // shift_wvt_vector(y,len,1,3,0);
  // shift_wvt_vector(y,len,2,3,0);
  // printf("\nLA8 transform - lifting inverse test");
  // res=lLA8(y,len,BWD,2);
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

  free(x);
  free(y);
  free(x1);
  
  printf("\n");
  return(0);
}
