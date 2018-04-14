#include "transform.h"
#include <math.h>

int main(void){
  uint len=8;
  int res=0, i;
  wst *w, *w2;
  real *x, *x2;
  uint minl, maxl;

  x = (real *)malloc(len*sizeof(real));
  x2 = (real *)malloc(len*sizeof(real));

  // read_1darray("noisydopC1024.csv",x,len);

  x[0]=0.; x[1]=0.; x[2]=5.; x[3]=4.;
  x[4]=8.; x[5]=6.; x[6]=7.; x[7]=3.;

  copyvec(x,x2,len);

  w = create_wvtstruct(x,MODWT_TO,HAAR,2,0,len);
  //w = create_wvtstruct(x,DWT,HAAR,2,0,len);
  
  w2 = create_wvtstruct(x2,MODWT_PO,HAAR,2,0,len);

  transform(w,FWD);
  transform(w2,FWD);

  // write_test_modwt(w->xmod,len,3);

  convert_modwt(w);

  res += cmpmodwt(w,w2,-1,10);

  transform(w,BWD);
  transform(w2,BWD);
  
  res += cmpvec(x,w->x,len);
  res += cmpvec(x,w2->x,len);
  
  if(res==0) printf("\nCongrats, no errors! :) \n");
  
  // print_modwt_vec_po(w->xmod,len,w->levels);
  // print_modwt_vec_to(w2->xmod,len,w2->levels);

  // cmpmodwt(w,w2,-1,10);
    
  kill_wvtstruct(w);
  kill_wvtstruct(w2);
  
  return(0);
}


// need to test all of the cmp modwt functions!
