#include "thresh.h"
#include "transform.h"
#include <math.h>

int main(void){
  real thresh;
  uint len=32;
  int res, i;
  wst *w, *w2;
  int n = 9;
  real *x, *x2;
  uint minl, maxl;

  x = (real *)malloc(len*sizeof(real));
  x2 = (real *)malloc(len*sizeof(real));

  read_1darray("noisydopC1024.csv",x,len);

  copyvec(x,x2,len);

  w = create_wvtstruct(x,MODWT_PO,HAAR,2,0,len);
  //w = create_wvtstruct(x,DWT,HAAR,2,0,len);
  
  w2 = create_wvtstruct(x2,MODWT_TO,HAAR,2,0,len);

  // transform(w,FWD);
  // transform(w2,FWD);

  // print_modwt_vec_po(w->xmod,len,w->levels);
  // print_modwt_vec_to(w2->xmod,len,w2->levels);

  // cmpmodwt(w,w2,-1,10);

  // // test median & mad functions
  // real* y;
  // y = (real *)malloc(n*sizeof(real));
  // real ymad, ymed;
  // for(i = 0; i < n; i++) y[i] = (double)i;
  // ymed = median(y,n);
  // ymad = mad(y,n);
  
  // // test interpolation
  // real ie[64], io[64];
  // for(i=0;i<64;i++){
  //   ie[i]=(double) 2*i;
  //   io[i]=(double) 2*i+1;
  // }

  // real tmp = interp_mse(x,ie,io,128);

  // printvec(x,len);

  //copyvec(x,xn,len);
  
  minl = 0;
  //  maxl = 1;
  maxl = w->levels-4;
  
  // res = transform(w,FWD);
  // thresh = univ_thresh(w,minl,maxl);
  // //thresh = 0.05;
  // threshold(w,NULL,thresh,SOFT,minl,maxl);
  // res = transform(w,BWD);

  printf("\n\nFirst CVT...\n");
  thresh=CVT(w,SOFT,0.01,minl,maxl);

  printf("\n\nSecond CVT...\n");
  thresh=CVT(w2,SOFT,0.01,minl,maxl);

  //printf("\ncalc univ thresh (using mean/var): %g",univ_thresh(x,len)); //not valid after running CVT as that writes thresholded vector to x!
  printf("\nthreshold is %g\n",thresh);
  
  write_1darray("Cthreshed.csv",w->x,len);
  
  kill_wvtstruct(w);
  kill_wvtstruct(w2);
  
  return(0);
}
