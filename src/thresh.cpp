#include "thresh.h"

void threshold_dwt(real* in, real*out, uint len, real thresh, short hardness){
  uint i;
  if(out==NULL) out = in;
  for(i=1;i<len;i++){
    // for all detail coefficients of a fully transformed vector...
    if(fabs(in[i])<thresh){
      // we threshold
      thresh_coef(out[i],thresh,hardness);
    }//endif
  }//endfor
}

void threshold_dwt(real* in, real* out, uint len, real thresh, short hardness, uint minlevel, uint maxlevel, uint levels){
  // NB level scaling is opposite from Nason
  uint i, skip;
  short cp = 0; // boolean to indicate whether we need to copy in to out
  // i.e. if in & out supplied, then cp = 0
  if(out==NULL) {out = in; cp = 1;}
  else out[0] = in[0]; //this is the top level scaling coeff. Not touched in rest of function.
  for(skip=1;skip<(1<<levels);skip=skip<<1){
    // printf("\nskip = %u;",skip);
    for(i=skip;i<len;i+=(skip<<1)){
      // looping through the detail coeffs of a DWT vector
      // printf("i=%u,",i);
      if((skip>=(1<<minlevel))&&(skip<=(1<<maxlevel))){
	out[i] = thresh_coef(in[i],thresh,hardness);
      }//if_skip_levels
      else{//else we don't threshold the levels - but we need to copy numbers across if different output vector
	if(cp==0) out[i]=in[i];
      }//else
    }//i_forloop
  }//skip_for_loop
}

void threshold_modwt(real* in, real* out, uint len, real thresh, short hardness, short modwttype, uint levels){
  threshold_modwt(in,out,len,thresh,hardness,modwttype,0,levels-1,levels);
}

void threshold_modwt(real* in, real* out, uint len, real thresh, short hardness, short modwttype, uint minlevel, uint maxlevel, uint levels){
  uint i, l, il;
  short cp = 1; // boolean to indicate whether we need to copy in to out
  short idone = 0; // boolean to indicate value thresholded or copied
  // i.e. if in & out supplied, then cp = 1
  if(out==NULL) {out = in; cp = 0;}
  for(l=0; l < levels; l++){
    // we loop through the levels
    il = 2*l*len; // base of loop counter
    for(i = 0; i<2*len; i++){
      // MODWT vectors have coefficients for level l between
      // il and il + 2*len
      if( (l >= minlevel) && (l<=maxlevel)){
	// if between min & max level then we have thresholding to do!	
	switch(modwttype){
	case MODWT_PO:
	  // detail coefficients are at odd indices
	  // from 2*l*len to 2*(l+1)*len
	  // (possibly covering multiple packets)
	  if((i%2) == 1){
	    out[il+i] = thresh_coef(in[il+i],thresh,hardness);
	    idone = 1;
	  }
	  break;
	case MODWT_TO :
	  // detail coefficients are at all indices
	  // from (2*l+1)*len to 2*(l+1)*len
	  if(i >= len){
	    out[il+i] = thresh_coef(in[il+i],thresh,hardness);
	    idone = 1;
	  }
	  break;
	default:
	  printf("\nUnrecognised modwt type %hi\n",modwttype);
	  break;  
	} //switch
      } // if min/max level
      if(cp && !idone) out[il + i] = in[il +i];
      // if we are copying & the value was not eligible for thresholding
      idone = 0;
    } // i loop
  } // l loop
}


void threshold(wst* win, wst* wout, real thresh, short hardness){
  if(win->ttype == DWT){
    if(wout==NULL){
      // if we are thresholding in place
      threshold_dwt(win->x,NULL,win->len,thresh,hardness);
    }
    else{
      // we are thresholding win & writing the result in wout
      threshold_dwt(win->x,wout->x,win->len,thresh,hardness);
      wout->transformed = 1;
      // we update the ouput vector to say it is transformed
    }
  }//if DWT
  if((win->ttype == MODWT_TO) || (win->ttype == MODWT_PO)){
    if(wout==NULL){
      // if we are thresholding in place
      threshold_modwt(win->xmod,NULL,win->len,thresh,hardness,win->ttype,win->levels);
    }
    else{
      threshold_modwt(win->xmod,wout->xmod,win->len,thresh,hardness,win->ttype,win->levels);
      wout->transformed = 1;
    // we update the ouput vector to say it is transformed
    }
  }// if MODWT
  
}

void threshold(wst* win, wst* wout, real thresh, short hardness, uint minlevel, uint maxlevel){
  if(check_len_levels(win->len,win->levels,minlevel,maxlevel,win->filtlen) > 0){
    // our len, flen, levels, minlevel & maxlevel are compatible!
    if(win->ttype == DWT){
      if(wout==NULL){
	// if we are thresholding in place
	threshold_dwt(win->x,NULL,win->len,thresh,hardness,minlevel,maxlevel,win->levels);
      }
      else{
	// we are thresholding win & writing the result in wout
	threshold_dwt(win->x,wout->x,win->len,thresh,hardness,minlevel,maxlevel,win->levels);
	wout->transformed = 1;
	// we update the ouput vector to say it is transformed
      }
    }//if DWT
    if((win->ttype == MODWT_TO) || (win->ttype == MODWT_PO)){
      if(wout==NULL){
	// if we are thresholding in place
	threshold_modwt(win->xmod,NULL,win->len,thresh,hardness,win->ttype,minlevel,maxlevel,win->levels);
      }
      else{
	threshold_modwt(win->xmod,wout->xmod,win->len,thresh,hardness,win->ttype,minlevel,maxlevel,win->levels);
	wout->transformed = 1;
	// we update the ouput vector to say it is transformed
      }
    }// if MODWT	
  }
}

real thresh_coef(real coef, real thresh, short hardness){
  if(fabs(coef) < thresh) coef = 0;
  else{
    // we don't threshold, but if SOFT thresholding
    // then we move coeff closer to 0 by thresh
    switch(hardness){
    case HARD:
      return(coef);
      break;
    case SOFT:
      return( ((coef < 0) ? -1 : 1)*(fabs(coef) - thresh));
      break;
    }
  }
  return(coef);
}


int null_level(real* x, uint len, short level){
  uint i;
  if((1 << level)>len) return(1); //not enough levels!
  int skip = 1 << level;
  for(i=skip-1;i<len;i+=skip){
    x[i]=0;
  }
  return(0);
}

real univ_thresh(wst*w, uint minlevel, uint maxlevel){
  real ut;
  real* tmp;
  uint l,n = 0;
  switch(w->ttype){
  case DWT:
    if((minlevel==0)&&(maxlevel==w->levels-1)){
      // then we can do this more efficiently without extra allocations
      // because we are only using detail coefficients
      // which are stored in w->x[1..len-1]
      return(mad(w->x+1,w->len-1)*sqrt(2.*log((double)(w->len-1))));
      break;
    }
    // otherwise, we need to allocate temporary vectors to work with
    for(l=minlevel;l<=maxlevel;l++) n+=(w->len)>>(l+1);
    // this calculates how big the tmp vector needs to be
    break;
  case MODWT_TO:
  case MODWT_PO:
    n = (maxlevel - minlevel + 1)*(w->len);
    break;
  default:
    printf("\nUnrecognised transform type\n");
    return(0);
    break;
  }
  tmp = (real *)malloc(n*sizeof(real));
  isolate_dlevels(w,minlevel,maxlevel,tmp,n);
  ut = mad(tmp,n)*sqrt(2.*log((double)(w->len)));
  free(tmp);
  return(ut);  
}

real CVT(wst*w, short hardness, real tol, uint minlevel, uint maxlevel){
  // w->x is original (noisy) vector

  real R = 0.61803399, C = 1. - R;

  real ta=0., tb, tc /*, ma=0., mb=0.,  mc=0.*/;
  wst *y1e, *y1o, *y2e, *y2o;
  wst *we, *wo, *wcpy;
  real t0, t1, t2, t3;
  real m1, m2;
  // ?a is lb, ?c is ub
  // ?0, ?1, ?2, ?3 denote ordered values that we are keeping track of
  // t? is threshold, m? is associated mse
  int res;
  uint len = w->len;
  uint lenh = len >> 1; // len/2
  uint maxlevel1;
  uint levels = w->levels;
  uint iter = 0;

  if(check_len_levels(w->len,w->levels,minlevel,maxlevel,w->filtlen) == 0){
    // error with levels
    return(0);
  }
  
  if(w->levels<=1){
    printf("\nNeed to be transforming at least one level for CVT!");
    return(0);
  }

  // allocate memory for auxiliary wavelet structs
  // these will be rewritten in the main loop
  y1e=create_wvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  y1o=create_wvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  y2e=create_wvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  y2o=create_wvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  
  // malloc w: we/o will hold un-thresholded wavelet wavelet coefficients of odd/even separated vectors
  we=create_wvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  wo=create_wvtstruct(w->ttype,w->filt,w->filtlen,w->levels-1,lenh);
  
  copyvecskip(w->x,2,len,we->x,1); // copy the even indices into we
  copyvecskip(w->x+1,2,len,wo->x,1); // copy the odd indices into wo

  // first calculate univ thresh value
  // ### should be transforming first then calculating threshold! ###
  wcpy = dup_wvtstruct(w); // we keep a copy of (untransformed) w to avoid extra transforms
  res=transform(w,FWD);
  tc = univ_thresh(w,minlevel,maxlevel);

  tb=tc/2.; // arbitrary threshold between ta & tc
  //threshold(x,y,len,tc,hardness);

  // makes points t2 & t1 st t0 to t1 is the smaller segment
  t0=ta;
  t2=tb;
  t1=tb - C*(tb - ta);
  t3=tc;
  
  maxlevel1 = maxlevel -1;

  // (forward) transform w vectors
  res=transform(we,FWD);
  res=transform(wo,FWD);
  
  //threshold using t1 & t2 
  threshold(we,y1e,t1,hardness, minlevel, maxlevel1);
  threshold(wo,y1o,t1,hardness, minlevel, maxlevel1);

  threshold(we,y2e,t2,hardness, minlevel, maxlevel1);
  threshold(wo,y2o,t2,hardness, minlevel, maxlevel1);

  // (inverse) transform y to x_hat (where x_hat is stored in y!)
  res=transform(y1e,BWD);
  res=transform(y1o,BWD);
  res=transform(y2e,BWD);
  res=transform(y2o,BWD);

  // calculate mse of interp(x_hat) vs xn for each of y1 & y2
  m1 = interp_mse(wcpy, y1e, y1o);
  m2 = interp_mse(wcpy, y2e, y2o);
    
  // minimise MSE by golden search...

  while(fabs(t3-t0) > tol*(t1+t2)){
    if(iter>50){
      printf("\nWe probably aren't converging. Exiting...\n");
      break;
    }
    // printf("\nt0 = %g, t1 = %g, t2 = %g, t3 = %g",t0,t1,t2,t3);
    // printf("\nm1 = %g, m2 = %g",m1,m2);
    if(m2 < m1){
      // m2 is new curr min, rearrange held points
      t0 = t1; t1=t2; t2=R*t2 + C*t3;
      m1=m2;
      threshold(we,y2e,t2,hardness,minlevel,maxlevel1);
      threshold(wo,y2o,t2,hardness,minlevel,maxlevel1);
      res=transform(y2e,BWD);
      res=transform(y2o,BWD);
      m2=interp_mse(wcpy,y2e,y2o);
    }
    else{
      t3=t2; t2=t1; t1=R*t1 + C*t0;
      m2=m1;
      threshold(we,y1e,t1,hardness,minlevel,maxlevel1);
      threshold(wo,y1o,t1,hardness,minlevel,maxlevel1);
      res=transform(y1e,BWD);
      res=transform(y1o,BWD);
      m1=interp_mse(wcpy,y1e,y1o);
    }
    iter++;
  }
  
  // loop: if not achieved tolerance
  //   thresh x with new t into y
  //   calc new MSE of interp x_hat vs xn
  
  // lastly - we will have to:
  //  - calculate thresh from scaling
  //  - threshold x with this thresh
  
  tc = m1<m2? t1 : t2;
  // tc now contains the chosen threshold
  tc = tc/(sqrt(1. - log(2.)/log((double)len)));
  // scale the threshold to the original, full, data

  threshold(w,NULL,tc,hardness,minlevel,maxlevel);
  res = transform(w,BWD);
  
  kill_wvtstruct(y1e);
  kill_wvtstruct(y1o);
  kill_wvtstruct(y2e);
  kill_wvtstruct(y2o);
  kill_wvtstruct(we);
  kill_wvtstruct(wo);
  kill_wvtstruct(wcpy);

  // printf("\n");
  return(tc);
}

real interp_mse(wst* wn, wst* wye, wst* wyo){
  // calculates interpolation error
  // comparing noisy wn to
  // smoothed ye & yo
  // i.e. interpolate ye & compare to odd values in w
  // & yo with even w
  uint i, j=0; //i is our counter in x, j counts in ye & yo
  real m=0., yie, yio; //m is mse, yie(o) is interp y[i] (y[i+1])
  uint len = wn->len;
  uint lenh = len/2;
  real *xn, *ye, *yo;
  
  xn = wn->x;
  ye = wye->x;
  yo = wyo->x;
  
  /* NB Nason's function WaveletCV interpolates the other way */
  /* It interpolates the noisy data to compare with the odd/even vector */
  
  for(i=0;i<len;i+=2){
   
    // We interpolate
    if(i==0){
      yie = yo[0];
      yio = 0.5*(ye[j] + ye[j+1]);
    }
    else
      if(i==len-2){
      yio = ye[lenh-1];
      yie = 0.5*(yo[j] + yo[j-1]);

    }
    else{
      yio = 0.5*(ye[j] + ye[j+1]);
      yie = 0.5*(yo[j] + yo[j-1]);
    }
    j++;
    m += (yie-xn[i])*(yie-xn[i]) + (yio-xn[i+1])*(yio-xn[i+1]); //update mse
  }
  return(m*0.5);///sqrt((double)len));
  //nason divides by 2 in his function.
}
