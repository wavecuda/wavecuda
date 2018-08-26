#include "wavutils.h"

int rejig(real* x, uint len, short int sense){
  real *tmp;
  uint skip=1, place=0, i, imax;
  // skip is for interleaving, place is placeholder in coalesced memory
  // i is counter, imax is ub of i in loops
  tmp=(real *)malloc(len*sizeof(real));
  copyvec(x,tmp,len);

  //first we copy the detail coeffs
  for(imax = len/2; imax >= 1; imax/=2){
    for(i=0;i<imax;i++){
      switch(sense){
      case CTI:	
	x[2*i*skip+skip] = tmp[i+place];
	// printf("we're setting x[%u] = tmp[%u]\n",2*i*skip+skip,i+place);
	break;
      case ITC:
	x[i+place] = tmp[2*i*skip+skip];
	// printf("we're setting x[%u] = tmp[%u]\n",i+place,2*i*skip+skip);
	break;
      default:
	printf("\nSense must be 0 (ITC) for Interleaved->Coalesced or 1 (CTI) for the reverse. For anything else, write your own script.\n");
	return(1);
      } 
    }
    place+=imax;
    skip=skip<<1;
  }

  //lastly, we copy the scaling coeff
  switch(sense){
  case CTI:	
    x[0]=tmp[len-1];
    //printf("we're setting x[%u] = tmp[%u]\n",0,len-1);
    break;
  case ITC:
    x[len-1]=tmp[0];
    //printf("we're setting x[%u] = tmp[%u]\n",len-1,0);
    break;
  }

  free(tmp);
  return(0);
}

uint check_len_levels(uint len, uint nlevels, uint filterlength){
  // in the case of different fwd/bwd filter lengths, filterlength should take the larger (provided you want to want to do both a fwd and bwd transform). This is just used to calculate the maximum possible level of transform.
  if(!((len != 0) && ((len & (~len + 1)) == len))){
    printf("\nLength %i is not a power of 2: ",len);
    printf("\nso we exit. Try again...\n");
    return(0);}
  // 0 is our error return
  uint maxlevels = (uint)floor(log2((double)len/filterlength))+1;
  // 'floor' is because filterlength isn't necessarily a power of 2
  if(nlevels==0) nlevels = maxlevels;
  if(nlevels>maxlevels){
    printf("\nnlevels, %u, is too large; this must be <= %i as vector length is %u amd filter length is %u",nlevels,maxlevels,len,filterlength);
    printf("\nso we exit. Try again...\n");
    return(0);
    // 0 is our error return
  }
  return(nlevels);
}

uint check_len_levels(uint len, uint nlevels, uint minlevel, uint maxlevel, uint filterlength){
  uint ret;
  ret = check_len_levels(len,nlevels,filterlength);
  if(ret==0) return(0); // 0 is an error return
  if((minlevel > maxlevel) || (maxlevel >= ret )){
    printf("\n##Error in given min/max levels: must be between 0 and %u ##\n",nlevels-1);
    printf("minlevel is %u, maxlevel is %u\n",minlevel,maxlevel);
    return(0);
  }
  return(ret);
}

void printwavlevel(real* x, uint len, uint level){
  //prints detail coefficients at specified level
  //in a DWT vector!
  uint skip,skip2,i;
  skip = 1<<level;
  skip2 = skip<<1;
  if(skip>=len) printf("\nLevel too large\n");
  printf("\nPrint wavelet coefficients, level %u...\n",level);
  for(i=skip;i<len;i+=skip2) printf("%g\n",x[i]);
  printf("\n");
}

void shift_wvt_vector(real* x, uint len, uint skip, int dshift, int sshift){
  // periodically shift a wavelet vector
  // to change time alignment
  // we are shifting coefficients right by 'shift'
  uint i, di, si, sai, dai;
  // i is loop iterator
  // d/si are detail/scaling specific loop iterators
  // d/sai are iterators inside temp detail/scaling array
  uint udshift = (dshift>=0)? dshift : -dshift; // abs value of shift
  uint usshift = (sshift>=0)? sshift : -sshift; // abs value of shift
  int dsign = (dshift==0)? 1 : dshift/((int)udshift);
  int ssign = (sshift==0)? 1 : sshift/((int)usshift);
  real* darray;
  real* sarray;
  darray=(real *)malloc((udshift+1)*sizeof(real));
  sarray=(real *)malloc((usshift+1)*sizeof(real));
  
  // loop inside vector & shifts
  // initial conditions of loop
  // here, we preload first values of temporary arrays
  for(sai=0;sai<=usshift;sai++){
    si = ssign==1 ? skip*(2*sai) : len - skip*(2*sai+2);
    // x index different for forward/backward shift
    sarray[sai]=x[si];
  }
  for(dai=0;dai<=udshift;dai++){
    di = dsign==1 ? skip*(2*dai+1) : len - skip*(2*dai+1);
    // x index different for forward/backward shift
    darray[dai]=x[di];
  }
  sai = 0;
  dai = 0;

  for(i=0; i<len; i+=2*skip){
    si = ssign==1 ? i : len-i-2*skip;
    // if ssign ==1, sshift is forwards, so loop forwards
    // if ssign ==-1, sshift is backwards, so loop backwards
    shift_loop(x,len,skip,usshift,ssign,si,sarray,&sai);
    // shift scaling coeffs
    di = dsign==1 ? (i+skip) : (len-(i+skip));
    // if dsign ==1, dshift is forwards, so loop forwards
    // if dsign ==-1, dshift is backwards, so loop backwards
    shift_loop(x,len,skip,udshift,dsign,di,darray,&dai);
    // shift detail coeffs
  }

  // switch(ssign + 2*dsign){
  //   //little optimisation avoiding doing extra loops
  // case -3:
  //   //ssign=-1, dsign=-1
  //   //loop both backwards
  //   for(i=len-2*skip;i>=0;i-=2*skip){
  //     shift_loop(x,len,skip,usshift,ssign,i,sarray,&si);
  //     // shift scaling coeffs
  //     shift_loop(x,len,skip,udshift,dsign,i+skip,darray,&di);
  //     // shift detail coeffs
  //   }
  //   break;

  // case -1:
  //   //ssign=1, dsign=-1
  //   //loop s forwards, d backwards
  //   for(i=len-2*skip;i>=0;i-=2*skip){
  //     shift_loop(x,len,skip,udshift,dsign,i+skip,darray,&di);
  //     // shift detail coeffs
  //   }
  //   for(i=0; i<len; i+=2*skip){
  //     shift_loop(x,len,skip,usshift,ssign,i,sarray,&si);
  //     // shift scaling coeffs
  //   }
  //   break;

  // case 1:
  //   //ssign=-1, dsign=1
  //   //loop s backwards, d forwards
  //   for(i=len-2*skip;i>=0;i-=2*skip){
  //     shift_loop(x,len,skip,usshift,ssign,i,sarray,&si);
  //     // shift scaling coeffs
  //   }
  //   for(i=0; i<len; i+=2*skip){
  //     shift_loop(x,len,skip,udshift,dsign,i+skip,darray,&di);
  //     // shift detail coeffs
  //   }
  //   break;

  // case 3:
  //   //ssign=1, dsign=1
  //   //loop both forwards
  //   for(i=0; i<len; i+=2*skip){
  //     shift_loop(x,len,skip,usshift,ssign,i,sarray,&si);
  //     // shift scaling coeffs
  //     shift_loop(x,len,skip,udshift,dsign,i+skip,darray,&di);
  //     // shift detail coeffs
  //   }
  // }

  free(darray);
  free(sarray);
}

void print_modwt_vec_po(real* x, uint len, uint nlevels){
  // print a packet-ordered wavelet vector
  uint lenos, cstart, skip, shift;
  uint l2s=0; // log_2 (skip)
  uint i;
  uint cw = (nlevels > MINCOL? nlevels: MINCOL) +2;
  // cw is just the column width for printing
  char ab0[cw+1];
  char ab1[cw+1];
  // a/b binary representations of the shift
  // to represent the shift of each packet
  
  
  printf("\nPrinting a packet ordered MODWT vector");

  for(skip=1;skip<(1<<nlevels);skip=skip<<1){
    lenos = len/skip;
    
    printf("\nLevel %u --------------------------------------",l2s+1);
    
    for(shift=0; shift<(skip<<1);shift+=2){
      cstart = 2*len*l2s + shift*lenos;
      
      ab_bin(ab0,shift,l2s,cw); // write to the ab0 string
      ab_bin(ab1,shift+1,l2s,cw); // write to the ab1 string
      
      printf("\n %*s  %*s\n",cw,ab0,cw,ab1);

      
      for(i=0;i<lenos;i++){
	printf("% *e % *e\n",cw,x[cstart+i],cw,x[cstart+lenos+i]);
      }
      printf("\n");
    }
   
    l2s++;
  }
  
}

void print_modwt_vec_to(real* x, uint len, uint nlevels){
  uint dstart, sstart;
  uint l;
  uint i;
  uint cw = (nlevels > MINCOL? nlevels: MINCOL) +2;
  // cw is just the column width for printing
  
  printf("\nPrinting a time ordered MODWT vector");

  for(l=0;l<nlevels;l++){
    
    printf("\nLevel %u --------------------------------------",l+1);
    
    sstart = len * 2 * l;
    dstart = len * (2 * l + 1);
    
    printf("\n %*s  %*s\n",cw,"s",cw,"d");
    
    for(i=0;i<len;i++){
      printf("% *e % *e\n",cw,x[sstart+i],cw,x[dstart+i]);
    }
    printf("\n");
    
  }
  
}

void ab_bin(char *ab, uint shift, uint l2s, uint cw){
  //fills strings with "a", "b", "aa", "ab" etc
  uint j;
  ab[cw] = '\0'; // the terminating character
  ab[cw-1] = '"'; // the end quote
  for(j=1; j<=cw-1; j++){
    if(j<=l2s+1){
      ab[cw-1-j] = shift & 1 ? 'b' : 'a';
      // 'b' if the last bit is 1, 'a' if it's 0
      shift = shift >> 1;
    }
    else{
      if((ab[cw-j]=='a')||(ab[cw-j]=='b')) ab[cw-1-j] = '"';
      else ab[cw-1-j] = ' ';
    }
  }
  
}
  

void shift_loop(real* x, uint len, uint skip, uint ushift, int sign, uint i, real* array, uint* ai){
  if(ushift>0){
    uint shi = (i + sign*ushift*2*skip) % len;
    // update next x index to be written to
    x[shi]=array[*ai];
    // write x index a shift away with what
    // used to be in x[i]
    array[*ai]=x[(shi+2*sign*skip) % len];
    // store next shift of x in temp array
    *ai = (*ai + 1) % (ushift+1);
  }
}

void ax_wvt_vector(real *x, uint len, uint skip, real da, real sa){
  uint i;
  // multiply coefficients of a wavelet vector by scalars
  // for a particular value of skip
  // da for the details
  // sa for the scalings
#pragma omp parallel for private (i)
  for(i=0;i<len;i+=(skip<<1)){
    x[i] = x[i] * sa;
    x[i+skip] = x[i+skip] * da;
  }
}

void writeavpackets(real* xr, real* xp0, real* xp1, uint len, uint skipxr){
  // this function works out the average of scaling coefficiens
  // in xp0 & xp1 - which are assumed to contain purely scaling coeffs
  // & we write into xr, which is assumed to contain interleaved detail/scaling
  // if we have not done any thresholding, then each scaling coefficient
  // should be equal to another all 3 vectors

  // this is clearly for the packet-ordered transform

  // the length of the packet at level j-1, xr, is 2*len,
  // whilst the length of the packets at level j, xp0 & xp1, is len
  
  // skipxr is 2 except for the final level of reconstruction,
  // in which we simply copy across coefficients
  
  uint i=0, i2=0;
  //#pragma omp parallel for private (i,i2)
  // we comment out the omp parallel stuff, as
  // this function is run from within an omp loop
  // in the fastest implementation
  for(i=0 ;i<len;i++){
    // i & i2 are initialised above
    //i2 = i*skipxr;
    xr[i2] = 0.5*(xp0[i] + xp1[(i-1) % len]);
    
    i2+=skipxr;
  }
  
}

void write_test_modwt(real* xm, uint len, uint nlevels){
  uint big10 = 10;
  while(big10 < len) big10*=10;
  uint l, i;
  
  for(l = 0; l < nlevels; l++){
    for(i = 0; i < len; i++){
      xm[2*l*len + i] = (double)big10*(l+1)+i;
      xm[(2*l+1)*len + i] = -(double)(big10*(l+1)+i);
    }
  }
}

int cmpmodwtlevelto(real* v1, real* v2, uint len, uint level, int numerrors){
  // now defunct! Use cmpdmowt instead
  uint serr, derr;
  uint sstart, dstart;
  
  sstart = len * 2 * level;
  dstart = len * (2 * level + 1);
  printf("\nTime-ordered MODWT vector, level %u",level);
  printf("\nScaling coefficients:\n");
  serr = cmpvec(v1 + sstart, v2 + sstart, len,1e-5,1e-10,numerrors);
  if(serr == 0) printf("\nno errors!");
  else printf("starting at %u\n",sstart);
  
  printf("\nDetail coefficients:\n");
  derr = cmpvec(v1 + dstart, v2 + dstart, len,1e-5,1e-10,numerrors);
  if(derr == 0) printf("\nno errors!");
  else printf("starting at %u\n",dstart);
  
  printf("\n");
  return(serr + derr);
  
}

int cmpmodwtlevelto(real* v1, real* v2, uint len, int numerrors){
  uint serr, derr;
  printf("\nTime-ordered MODWT vector");
  printf("\nScaling coefficients:\n");
  serr = cmpvec(v1, v2, len,1e-5,1e-10,numerrors);
  if(serr == 0) printf("no errors!");
  
  printf("\nDetail coefficients:\n");
  derr = cmpvec(v1+len, v2+len, len,1e-5,1e-10,numerrors);
  if(derr == 0) printf("no errors!");
  
  printf("\n");
  return(serr + derr);

}

int cmpmodwtlevelpo(real* v1, real* v2, uint len, uint l, int numerrors){
  // for PO, we need to know level information
  // to give information about packets
  
  uint skip, lenos;
  uint shift, npackets;
  uint cw = l + 3; // string width for printing packet ab info
  char ab[cw + 1]; // +1 for EOF string end
  uint sderr=0, perr;
  
  skip = (1<<l);
  lenos = len/skip;
  npackets = skip<<1;

  printf("\nPacket-ordered MODWT vector\n");
  
  for(shift=0; shift < npackets; shift++){
    perr = cmpvec(v1 + shift*lenos, v2 + shift*lenos, lenos,1e-5,1e-10,numerrors);
    if((perr > 0) && (numerrors > 0)){
      // we print info on the errors
      ab_bin(ab,shift,l,cw);
      printf("--In packet %u, %s.\n\n",shift,ab);
    }
    sderr += perr;
    numerrors = (numerrors - (int)sderr) > 0 ? (numerrors - (int)sderr) : 0;
  }
  if(sderr == 0) printf("no errors!");
  return(sderr);
}



int cmpmodwt(wst* w1, wst* w2, int level, int numerrors){
  uint newerrors, toterr=0;
  uint sstart, dstart;
  real *sd1, *sd2, *sd_to;
  uint len = w1->len;
  wst *w;
  int k;
  uint l, minl, maxl;
  uint skip, lenos;
  uint shift, shift_to, npackets;
  short t1 = w1->ttype, t2 = w2->ttype;
  int res;
  
  if(!( ((t1 == MODWT_TO)||(t1 == MODWT_PO)) && ((t1 == MODWT_TO)||(t1 == MODWT_PO)) )){
    printf("\nNot of MODWT type!\n");
    return(1);
  }
  
  if(w1->len != w2->len){
    printf("\nCan't compare MODWT wst types with different x lengths\n");
    return(1);
  }
  
  if (level < 0){
    minl = 0;
    maxl = w1->levels < w2->levels ? ((w1->levels) - 1) : ((w2->levels) - 1);
    // if they have different levels of transform
    // we only check up to the min of the two
  }
  else{
    if((level > (w1->levels) - 1) || (level > (w2->levels) - 1)){
      printf("\nError: level %d given is too high: should be <= %u and %u\n",level,w1->levels,w2->levels);
      return(1);
    }
    minl = level;
    maxl = level;
  }
  
  // need to test what both ttypes are

  // TO & TO - run TO checker
  // PO & PO - run PO checker
  // TO & PO - convert PO to TO & run TO checker

  // 1 level at a time
  
  if(t1 != t2){
    // then we have 1 TO and 1 PO,
    // so we will have to do some converting
    // we allocate an array to
    sd_to = (real *) malloc(2*len*sizeof(real));
  }

  for(l = minl; l <= maxl; l++){
    printf("\nLevel %u --------------------------------------",l+1);
    sstart = len * 2 * l;
   
    if(t1==t2){
      // both inputs are of the same type of MODWT transform
      if(t1==MODWT_TO){	
	// we are comparing time-ordered vectors
	sd1 = w1->xmod + sstart;
	sd2 = w2->xmod + sstart;
	newerrors = cmpmodwtlevelto(sd1,sd2,len,numerrors);
      }
      else{
	// we are comparing packet-ordered vectors
	sd1 = w1->xmod + sstart;
	sd2 = w2->xmod + sstart;
	newerrors = cmpmodwtlevelpo(sd1,sd2,len,l,numerrors);
      }
    }
    else{
      // we have different types of MODWT transform
      printf("\nWe are comparing a different types of MODWT transform.");
      if(t1==MODWT_PO){
	printf(" We will convert a copy of the first input to TO for comparison.");
	// we convert w1 to TO type
	// and put this output into sd_to
	// with sd1 pointed to sd_to
	sd1 = sd_to;
	res = convert_modwt_level(w1->xmod + sstart,sd1,l,MODWT_PO,len);
	sd2 = w2->xmod + sstart;
      }
      else{
	printf(" We will convert a copy of the second input to TO for comparison.");
	// we convert w2 to TO type
	// and put this output into sd_to
	// with sd2 pointed to sd_to
	sd1 = w1->xmod + sstart;
	sd2 = sd_to;
	res = convert_modwt_level(w2->xmod + sstart,sd2,l,MODWT_PO,len);
      }
      newerrors = cmpmodwtlevelto(sd1,sd2,len,numerrors);
    }

    toterr += newerrors;
    numerrors = (numerrors - (int)newerrors) > 0 ? (numerrors - (int)newerrors) : 0;
    // numerrors controls how many errors are printed by cmpvec
    // negative value means print all errors
    // so we reduce it by newerrors but ensure the answer is >=0
    
  }// l loop
  
  if(t1 != t2){
    // then we had 1 TO and 1 PO,
    // and allocated an array
    free(sd_to);
  }
  
  printf("\n");
  return(toterr);
}

int convert_modwt(wst* w){
  // function converts a modwt wst structure
  // from PO -> TO
  // or from TO -> PO
  // depending on the type of the input
  uint l, minl, maxl;
  real* tmp;
  uint sstart;
  int res;
  short wtype = w->ttype;
  uint len = w->len;

  if( (w->levels == 0) || (w->transformed == 0) ){
    printf("\nWavelet input not transformed. We just need to change the MODWT type.\n");
    w->ttype = (wtype == MODWT_TO) ? MODWT_PO : MODWT_TO;
    return(0);
  }
  
  tmp = (real *) malloc(2*len*sizeof(real));
  
  minl = 0;
  maxl = (w->levels) - 1;

  for(l = minl; l <= maxl; l++){
    sstart = len * 2 * l;
    res = convert_modwt_level(w->xmod + sstart,tmp,l,wtype,len);
    if(res == 1) return(1);
    copyvec(tmp,w->xmod + sstart,2*len);
  }
  w->ttype = (wtype == MODWT_TO) ? MODWT_PO : MODWT_TO;
  printf("\nConverted from %s to %s MODWT\n",
	 (wtype == MODWT_TO) ? "time-ordered" : "packet-ordered",
	 (w->ttype == MODWT_TO) ? "time-ordered" : "packet-ordered");
  free(tmp);
  return(0);
}

int convert_modwt_level(real* xin, real* xout, uint l, short intype, uint len){
  // converts a vector of a level of 'len' modwt coefficients
  // from intype to not intype
  uint skip, lenos;
  uint shift, shift_to, npackets;
  
  skip = (1<<l);
  lenos = len/skip;
  npackets = skip<<1;

  for(shift=0; shift < npackets; shift++){
    shift_to = reverse_bits(shift,l+1);
    // the time order of the packets turns out to be the given by reversing
    // the bits of the shift values
    // cool, huh?!
    
    switch(intype){
    case MODWT_PO:
      copyvecskip(xin + shift*lenos,2,lenos,xout + shift_to,2*skip);
      copyvecskip(xin + 1 + shift*lenos,2,lenos,xout + len + shift_to,2*skip);
      break;
    case MODWT_TO:
      copyvecskip(xin + shift_to,2*skip,len,xout + shift*lenos,2);
      copyvecskip(xin + len + shift_to,2*skip,len,xout + 1 + shift*lenos,2);
      break;
    default:
      printf("\nnot a MODWT type!\n");
      return(1);
    }// switch
  }// shift loop
  return(0);
}

wst* create_wvtstruct(short ttype, short filt, short filtlen, uint levels, uint len){
  double *x;
  levels = check_len_levels(len,levels,filtlen);
  if(levels == 0) return(NULL);
  // this is an error trap for incompatible level/len/filtlen
  x=(real *)malloc(len*sizeof(real));
  
  return(create_wvtstruct(x,ttype,filt,filtlen,levels,len));
}

wst* create_wvtstruct(real* x, short ttype, short filt, short filtlen, uint levels, uint len){
  real *xmod;
  wst* w = (wst *)malloc(sizeof(wst));
  levels = check_len_levels(len,levels,filtlen);
  if(levels == 0) return(NULL);
  // this is an error trap for incompatible level/len/filtlen
  if((ttype == MODWT_TO) || (ttype == MODWT_PO))
    xmod=(real *)malloc(len*2*levels*sizeof(real));
  // if the transform type is MODWT, then we need to
  // allocate the modwt vector
  w->x = x;
  w->ttype = ttype;
  w->filt = filt;
  w->filtlen = filtlen;
  w->transformed = 0;
  w->levels = levels;
  w->len = len;
  w->xmod = xmod;
  
  return(w);
}

wst* create_wvtstruct(real* x,real* xmod, short ttype, short filt, short filtlen, uint levels, uint len){
  wst* w = (wst *)malloc(sizeof(wst));
  levels = check_len_levels(len,levels,filtlen);
  if(levels == 0) return(NULL);
  // this is an error trap for incompatible level/len/filtlen
  w->x = x;
  w->ttype = ttype;
  w->filt = filt;
  w->filtlen = filtlen;
  w->transformed = 0;
  w->levels = levels;
  w->len = len;
  w->xmod = xmod;
  
  return(w);
}


wst* dup_wvtstruct(wst *w1, short memcpy){
  wst *w2;
  w2 = create_wvtstruct(w1->ttype,w1->filt,w1->filtlen,w1->levels,w1->len);
  // allocates wvtstruct & sets non-pointer components

  if(memcpy){
    // then we copy the array elements across too
    // (we avoid doing this in cross validation when we don't have to)
    copyvec(w1->x,w2->x,w1->len);
    if((w1->ttype == MODWT_TO) || (w1->ttype == MODWT_PO))
      copyvec(w1->xmod,w2->xmod,(w1->len)*2*(w1->levels));
  }
  return(w2);
}

wst*  dup_wvtstruct(wst *w1){
  // this behaviour duplicates with memcpy of pointer components
  return(dup_wvtstruct(w1,1));
}

void kill_wvtstruct(wst *w){
  free(w->x);
  if((w->ttype == MODWT_TO) || (w->ttype == MODWT_PO))
    free(w->xmod);
  free(w);
}

void remove_wvtstruct(wst *w){
  free(w);
}

void isolate_dlevels(wst* w, uint minlevel, uint maxlevel, real* dvec, uint n){
  uint i = 0,j=0, l, il;
  uint skip, levels, len;
  uint minskip, maxskip;
  len = w-> len;
  levels = w->levels;
  switch(w->ttype){
  case DWT:
    minskip= (1<<minlevel);
    maxskip = (1<<maxlevel);
    j = 0; // making sure!
    for(skip=1;skip<(1<<levels);skip=skip<<1){
      for(i=skip;i<len;i+=(skip<<1)){
	// looping through the detail coeffs of a DWT vector
	if((skip>=minskip)&&(skip<=maxskip)){
	  if(j<n){
	    dvec[j] = w->x[i];
	    j++;
	  }
	  else{
	    printf("\nError in n calculation!\n");
	    break;
	  }
	}
      }
    }
    if(j != n) printf("\nError in n calculation!\n");
    break;
  case MODWT_TO:
    il = len;
    for(l = minlevel; l <= maxlevel; l++){
      copyvec(w->xmod + il,dvec + j,len);
      j+=len;
      il+=2*len;
    }
    break;
  case MODWT_PO:
    il = 0;
    for(l = minlevel; l <= maxlevel; l++){
      copyvecskip(w->xmod + il + 1,2,len*2,dvec + j,1);
      j+=len;
      il+=2*len;
    }    
    break;
  default:
    printf("\nUnrecognised transform type\n");
    break;
  }

}

void print_wst_info(wst *w){
  printf("\n-------------------------------");
  printf("\nCPU wavelet structure");
  printf("\n-------------------------------");
  printf("\nFilter: ");
  switch(w->filt){
  case HAAR: 
  case HAARMP:
    printf("Haar"); break;
  case DAUB4:
  case DAUB4MP:
    printf("Daub4"); break;
  default:
    printf("Unknown filter"); break;
  }
  printf("\nTransform type: ");
  switch(w->ttype){
  case DWT: printf("DWT"); break;
  case MODWT_TO: printf("MODWT, time ordered"); break;
  case MODWT_PO: printf("MODWT, packet ordered"); break;
  default:
    printf("Unknown transform type"); break;
  }
  printf("\nLevels: %u",w->levels);
  printf("\nLength: %u",w->len);
  printf("\n-------------------------------");
  printf("\n");
}

uint ndetail_thresh(short ttype, uint len, uint minlevel, uint maxlevel){
  uint n_d=0;
  uint skip;
  switch(ttype){
  case DWT:
    for(skip = (1<<minlevel); skip <= (1<<maxlevel); skip = skip<<1){
      n_d += len/(skip<<1);
    }
    break;
  case MODWT_TO:
  case MODWT_PO:
    n_d = (maxlevel - minlevel + 1) * len;
    break;
  }
  return(n_d);
}

uint ndetail_thresh(wst* w, uint minlevel, uint maxlevel){
  return(ndetail_thresh(w->ttype,w->len,minlevel,maxlevel));
}

short get_filt_len(short filt){
  switch(filt){
  case HAAR:
  case HAARMP:
    return(2);
    break;
  case DAUB4:
  case DAUB4MP:
    return(4);
    break;
  case C6F:
  case C6FMP:
    return(6);
    break;
  case LA8F:
  case LA8FMP:
    return(8);
    break;
  default:
    return(0);
    break;
  }
}
