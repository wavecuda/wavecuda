#include "rfunctions.cuh"

void RcpuTransform(real* x, real* xmod, int* len, int* sense, int* nlevels, int * ttype, int* filter, int* filterlen){
  int res;
  wst *w;

  // pop the vectors into a wavelet structure, & away we go!
  
  w=create_wvtstruct(x,xmod,(short) *ttype,(short) *filter, (short) *filterlen, (uint) *nlevels, (uint) *len);
      
  if(*sense==BWD) w->transformed = 1;

  res = transform(w,*sense);

  remove_wvtstruct(w);
  
  if(res!=0) Rprintf("\n Error in CPU transform\n");
}


void RgpuTransform(real* x, real* xmod, int* len, int* sense, int* nlevels, int * ttype, int* filter, int* filterlen){
  int res;
  cuwst *w;

  // for the GPU setting, we need an extra memcopy
  // as CUDA prefers to allocate the host memory
  // to allow any stream behaviour
  
  w=create_cuwvtstruct((short) *ttype,(short) *filter, (short) *filterlen, (uint) *nlevels, (uint) *len);
  
  if( (*sense==BWD) && ((*ttype==MODWT_TO)||(*ttype==MODWT_PO)) )
    {
      copyvec(xmod,w->xmod_h,(*len)*2*(*nlevels));
    }
  else{
    copyvec(x,w->x_h,(uint) *len);
  }

  if(*sense==BWD) w->transformed = 1;
  
  res = transform(w,*sense);

  cudaDeviceSynchronize();
  
  if( (*sense==FWD) && ((*ttype==MODWT_TO)||(*ttype==MODWT_PO)) )
    {
      copyvec(w->xmod_h,xmod,(uint) (*len)*2*(*nlevels));
    }
  else{
    copyvec(w->x_h,x,(uint) *len);
  }
  
  kill_cuwvtstruct(w);
  
  if(res!=0) Rprintf("\n Error in CPU transform\n");
}


void RcpuThreshold(real* x, real* xmod, int* len, int* nlevels, int * ttype, int* filter, int* filterlen, real* thresh, int* hardness, int* minlevel, int* maxlevel){
  wst *w;

  // pop the vectors into a wavelet structure, & away we go!
  
  w=create_wvtstruct(x,xmod,(short) *ttype,(short) *filter, (short) *filterlen, (uint) *nlevels, (uint) *len);
  
  threshold(w,NULL,*thresh,(short) *hardness,*minlevel,*maxlevel);
  
  remove_wvtstruct(w);
}

void RcpuSmooth(real* x, int* len, int* nlevels, int * ttype, int* filter, int* filterlen, int* threshtype, real* thresh, int* hardness, int* minlevel, int* maxlevel, real* tol){
  wst *w;

  // pop the vectors into a wavelet structure, & away we go!
  
  w=create_wvtstruct(x,(short) *ttype,(short) *filter, (short) *filterlen, (uint) *nlevels, (uint) *len);

  switch(*threshtype){
  case CV:
    *thresh = CVT(w,(short) *hardness,*tol,*minlevel,*maxlevel);
    break;
  case UNIV:
    transform(w,FWD);
    *thresh = univ_thresh(w,0,0);
    threshold(w,NULL,*thresh,(short) *hardness,*minlevel,*maxlevel);
    transform(w,BWD);
    break;
  case MANUAL:
    transform(w,FWD);
    threshold(w,NULL,*thresh,(short) *hardness,*minlevel,*maxlevel);
    transform(w,BWD);
    break;
  default:
    Rprintf("\n Unknown threshold type\n");
    break;
  }
  
  remove_wvtstruct(w);
  
}

void RgpuSmooth(real* x, int* len, int* nlevels, int * ttype, int* filter, int* filterlen, int* threshtype, real* thresh, int* hardness, int* minlevel, int* maxlevel, real* tol){
  cuwst *w;

  // for the GPU setting, we need an extra memcopy
  // as CUDA prefers to allocate the host memory
  // to allow any stream behaviour

  w=create_cuwvtstruct((short) *ttype,(short) *filter, (short) *filterlen, (uint) *nlevels, (uint) *len);
  
  copyvec(x,w->x_h,(uint) *len);

  switch(*threshtype){
  case CV:
    *thresh=CVT(w,(short) *hardness, *tol, (uint) *minlevel, (uint) *maxlevel);
    break;
  case MANUAL:
    update_cuwst_device(w); //copies x_h to x_d or xmod_h to xmod_d
    w->filt = - (w->filt); // change transform to be nohost
    transform(w,FWD);
    cudaDeviceSynchronize();
    threshold(w,NULL,*thresh,(short) *hardness, (uint) *minlevel, (uint) *maxlevel,NULL);
    cudaDeviceSynchronize();
    transform(w,BWD);
    update_cuwst_host(w); //copies x_d to x_h or xmod_d to xmod_h
    break;
  default:
    Rprintf("\n Unknown threshold type\n");
    break;
  }
  
  cudaDeviceSynchronize();
  copyvec(w->x_h,x,(uint) *len);

  kill_cuwvtstruct(w);
  
}

SEXP RgpuTransformList(SEXP argslist){
  // expecting a list of lists
  // with each sublist containing x, xmod, len, filt etc
  int listlen = length(argslist);
  SEXP list1 = VECTOR_ELT(argslist,0); // first elt of argslist, itself a list
  real* x1 = REAL(VECTOR_ELT(list1,0)); // first elt of list1 is x vector
  
  int xlen = INTEGER(VECTOR_ELT(list1,2))[0];

  printf("\n***xlen is %d***\n",xlen);

  SEXP ans = allocVector(VECSXP,2);
  SET_VECTOR_ELT(ans,0,ScalarReal(x1[0]));
  SET_VECTOR_ELT(ans,1,ScalarInteger(xlen));
  return(ans);
}
