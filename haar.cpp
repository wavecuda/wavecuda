#include "haar.h"
#include "haarcoeffs.h"
#include "utils.h"

/*---------------------------------------------------------
  Functions for serial CPU wavelet transform
  ---------------------------------------------------------*/

/*
  data structure used is the following:
  x is our vector

  x                        transform
  i 
  0   | 0 |   |s11|   |s21|   |s31|
  1   | 0 |   |d11|   |d11|   |d11|
  2   | 5 |   |s12|   |d21|   |d21|
  3   | 4 |   |d12|   |d12|   |d12|
  4   | 8 |   |s13|   |s22|   |d31|
  5   | 6 |   |d13|   |d13|   |d13|
  6   | 7 |   |s14|   |d22|   |d22|
  7   | 3 |   |d14|   |d14|   |d14|

      skip=1  skip=2  skip=4  skip=8

  ----> forward transform ---->
  <--- backward transform <----

  where sij (dij) is the scaling (detail) coefficient number j at level i.
*/

int Haar(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaar(x,len,1,nlevels));
  case BWD:
    return(bHaar(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaar(real* x, uint len, uint skip, uint nlevels){
  real tmp;
  uint i;
  // static uint opsf = 0;
  // static uint loops = 0;
  /* double t; */
  //printf("\nenter the Haar\n");
  if(skip < (1 << nlevels)){
    // we have something to do
    /* timer(-1); */
    for (i=0;i<len;i+=2*skip){    
      tmp = (x[i] - x[i+skip])*invR2;
      x[i] = (x[i] + x[i+skip])*invR2;
      x[i+skip] = tmp;
      // opsf+=4;
      // loops+=1;
    }
    // printf("\nskip = %u",skip);
    // printf("\nsubloops = %u",loops);
    /* t=timer(1); */
    /* printf("skip = %i, t = %g\n",skip,t); */
    //    printf("Haarit (%g,%g,%g,%g,%g,%g,%g,%g)\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    // printf("Skip: %i\n",skip);
    return(fHaar(x,len,skip<<1,nlevels));
  }
  // printf("\nOpsf = %u",opsf);
  // printf("\nLoops = %u",loops);
  return(0);
}

int bHaar(real* x, uint len, uint skip){
  double tmp;
  uint i;
  //printf("\nenter the Haar\n");
  if(skip > 0){
    // we have something to do
    for (i=0;i<len;i+=2*skip){
      tmp = (x[i] - x[i+skip])*invR2;
      x[i] = (x[i] + x[i+skip])*invR2;
      x[i+skip] = tmp;
    }
    // printf("iHaarit (%g,%g,%g,%g,%g,%g,%g,%g)\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    // printf("Skip: %i\n",skip);
    return(bHaar(x,len,skip>>1));
  }
  return(0);
}

/*---------------------------------------------------------
  Functions for openmp (CPU) parallel wavelet transform
  ---------------------------------------------------------*/

/*
  Same memory as above
 */

int Haarmp(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case 1:
    return(fmpHaar(x,len,1,nlevels));
  case 0:
    return(bmpHaar(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fmpHaar(real* x, uint len, uint skip, uint nlevels){
  real tmp;
  uint i;
  /* double t; */
  //printf("\nenter the Haar\n");
  if(skip < (1 << nlevels)){
    // we have something to do
    /* mptimer(-1); */
#pragma omp parallel for private (i,tmp)
    for (i=0;i<len;i+=2*skip){
      /* int thread_id = omp_get_thread_num(); */
      /* printf("thread_id = %i, i = %i, i+skip = %i\n",thread_id,i,i+skip); */
      tmp = (x[i] - x[i+skip])*invR2;
      x[i] = (x[i] + x[i+skip])*invR2;
      x[i+skip] = tmp;
    }
    /* t=mptimer(1); */
    /* printf("skip = %i, t = %g\n",skip,t); */
    // printf("Haarit (%g,%g,%g,%g,%g,%g,%g,%g)\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    // printf("Skip: %i\n",skip);
    return(fmpHaar(x,len,skip<<1,nlevels));
  }
  return(0);
}

int bmpHaar(real* x, uint len, uint skip){
  real tmp;
  uint i;
  //printf("\nenter the Haar\n");
  if(skip > 0){
#pragma omp parallel for private (i,tmp)
    // we have something to do
    for (i=0;i<len;i+=2*skip){
      /* int thread_id = omp_get_thread_num(); */
      /* printf("thread_id = %i, i = %i, i+skip = %i\n",thread_id,i,i+skip); */
      tmp = (x[i] - x[i+skip])*invR2;
      x[i] = (x[i] + x[i+skip])*invR2;
      x[i+skip] = tmp;
    }
    // printf("iHaarit (%g,%g,%g,%g,%g,%g,%g,%g)\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    // printf("Skip: %i\n",skip);
    return(bmpHaar(x,len,skip>>1));
  }
  return(0);
}

/*---------------------------------------------------------
  Functions for serial CPU wavelet transform with coalesced memory
  ---------------------------------------------------------*/

/*
  data structure used is the following:
  x is our vector

  x                        transform
  i 
  0   | 0 |   |d11|   |d11|   |d11|
  1   | 0 |   |d12|   |d12|   |d12|
  2   | 5 |   |d13|   |d13|   |d13|
  3   | 4 |   |d14|   |d14|   |d14|
  4   | 8 |   |s11|   |d21|   |d21|
  5   | 6 |   |s12|   |d22|   |d22|
  6   | 7 |   |s13|   |s21|   |d31|
  7   | 3 |   |s14|   |s22|   |s31|

      pos=0   pos=4    p=6    pos=7

  ----> forward transform ---->
  <--- backward transform <----

  where sij (dij) is the scaling (detail) coefficient number j at level i.
*/


int HaarCoalA(real* x, uint len, short int sense){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  if(!((len != 0) && ((len & (~len + 1)) == len))){
    printf("\nLength %i is not a power of 2: ",len);
    printf("\nso we exit. Try again...\n");
    return(1);}
  switch(sense){
  case 1:
    return(fHaarCA(x,len,0));
  case 0:
    return(bHaarCA(x,len,len-1));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaarCA(real* x, uint len, uint pos){
  uint ssize = (len - pos)>>1;
  real s[ssize];
  uint i,j;
  //printf("\nenter the Haar\n");
  if(pos < (len-1)){
    // we have something to do
    
    //also try s = s - ssize & reduce additions!

    for (i=0,j=pos;i<ssize;i++,j+=2){
      printf("\ni = %u, j = %u, pos = %u",i,j,pos);
      s[i] = (x[j] + x[j+1])*invR2;
      x[i+pos] = (x[j] - x[j+1])*invR2;
    }
    for (i=0;i<ssize;i++){
      x[i+pos+ssize] = s[i];
    }
    free(s);
    // printf("\nsubloops = %u",loops);
    printf("Haarit (%g,%g,%g,%g,%g,%g,%g,%g)\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    return(fHaarCA(x,len,pos+ssize));
  }
  // printf("\nOpsf = %u",opsf);
  // printf("\nLoops = %u",loops);
  return(0);
}


// we expect the backwards transform to be not as fast due to larger memory allocations & less coalescedness.
int bHaarCA(real* x, uint len, uint skip){
  double tmp;
  uint i;
  //printf("\nenter the Haar\n");
  if(skip > 0){
    // we have something to do
    for (i=0;i<len;i+=2*skip){
      tmp = (x[i] - x[i+skip])*invR2;
      x[i] = (x[i] + x[i+skip])*invR2;
      x[i+skip] = tmp;
    }
    // printf("iHaarit (%g,%g,%g,%g,%g,%g,%g,%g)\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    // printf("Skip: %i\n",skip);
    return(bHaarCA(x,len,skip>>1));
  }
  return(0);
}


/*---------------------------------------------------------
  Functions for serial CPU packet-ordered MODWT
  ---------------------------------------------------------*/

/*
  data structure used is the following:
  x is our vector

  eg,

  x    = [x0    x1    x2    x3    x4    x5    x6    x7    ]

  which is transformed to vector xdat

  xdat = c( x1a  , x1b  , x2aa , x2ab , x2ba , x2bb ,
            x3aaa, x3aab, x3aba, x3abb, x3baa, x3bab, x3bba, x3bbb )

  where c( , ) is R concatenation notation and 
  
  x1a  = [s10   d10   s11   d11   s12   d12   s13   d13  ]
  is the first packet (with no shift) for level 1, i.e. s10 = (1/sqrt(2)) * (x[1] + x[0]), d10 = (1/sqrt(2)) * (x[1] - x[0])

  x1b  = [s10^1 d10^1 s11^1 d11^1 s12^1 d12^1 s13^1 d13^1]
  is the second packet for level 1, i.e. s10^1 = (1/sqrt(2)) * (x[2] + x[1])

  x2aa = [s20   d20   s21   d21   ]
  is first packet of level 2, i.e. s20 = (1/sqrt(2)) * (x1a[2] + x1a[0]) = (1/sqrt(2)) * (s10 + s11)

  x2ab = [s20^1 d20^1 s21^1 d21^1 ]
  is the second packet of level 2, s20^1 = (1/sqrt(2)) * (x1a[4] + x1a[2]) = (1/sqrt(2)) * (s11 + s12)

  x2ba = [s20^2 d20^2 s21^2 d21^2 ]
  is the third packet of level 2, s20^2 = (1/sqrt(2)) * (x1b[2] + x1b[0]) = (1/sqrt(2)) * (s10^1 + s11^1)
  
  x2bb = [s20^3 d20^3 s21^3 d21^3 ]
  is the fourth packet of levels 2, s20^3 = (1/sqrt(2)) * (x1b[4] + x1b[2]) = (1/sqrt(2)) * (s11^1 + s12^1)

  x3aaa= [s30 d30 ]
  is the first packet of level 3, s30 = (1/sqrt(2)) * (x2aa[2] + x2aa[0]) = (1/sqrt(2)) * (s20 + s21)

  and x3aab, x3aba, x3abb, x3baa, x3bab, x3bba, x3bbb defined similarly

  This version works by copying the "s" coefficieints to the next level of then transforming in place.
  
*/



int HaarMODWT(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarMODWT(x,xdat,len,1,nlevels));
  case BWD:
    return(bHaarMODWT(x,xdat,len,1<<(nlevels-1),nlevels));
    //    printf("\nNot yet implemented!\n");
    //    return(1);
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaarMODWT(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int shift, shift2, lenos, cstart, copyfrom;
  int l2s = 0; // log2(skip)
  int res;

  //  *xdat=(real *)malloc(len*2*nlevels*sizeof(real));

  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=skip<<1){
    lenos=len/skip; // we pre-compute this for efficiency
    
    //loop through shifts
    for(shift=0;shift<(skip<<1);shift++){
      shift2 = shift % 2;
      
      cstart = 2*len*l2s + shift*lenos;
      copyfrom = cstart - 2*len - shift2*lenos;
      // this means we copy the same scaling coefficients twice
      // for transforming in place with two shifts consecutively
      // we would get better time alignment with a slightly different
      // ordering, but this is packet ordering!
      
      if(skip==1){
	copyvecskipshiftred(x,1,len,xdat + cstart,shift2);
      }else{
	copyvecskipshiftred(xdat+copyfrom,2,lenos*2,xdat+cstart,shift2);
      }

      // do single level wvt transform on current level
      res = fHaar(xdat+cstart,lenos,1,1);

    }
    
    l2s++; //update log2(shift)
  }
  return(0);
}

int bHaarMODWT(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int shift, shift2, lenos, cstart, copyto;
  int l2s = nlevels-1; // log2(skip)
  int res;
    
  // loop over skip
  for( ; skip>=1; skip=skip>>1){
    // skip is already set before this function is run
    lenos=len/skip; // we pre-compute this for efficiency
      
    //loop through shifts
    for(shift=0;shift<(skip<<1);shift++){
      shift2 = shift % 2;
	
      cstart = 2*len*l2s + shift*lenos;
      copyto = cstart - 2*len - shift2*lenos;
	
      // do single level inv wvt transform on current packet
      res = bHaar(xdat+cstart,lenos,1);
      
      if(shift2==1){
	// we take the average of the two reconstructions
	if(skip>1){
	  // we stay in the xdat vector
	  writeavpackets(xdat+copyto,xdat+cstart-lenos,xdat+cstart,lenos,2);
	}
	else{
	  // we write the final reconstructed x values into the x vector
	  writeavpackets(x,xdat+cstart-lenos,xdat+cstart,lenos,1);
	}
      }
      
    }
    
    l2s--; //update log2(shift) 
  }
  return(0);
}



// this version is slow, so we didn't bother writing the BWD trans.
// this does packets serially, but each wvt trans runs an omp version
int HaarMODWTomp1(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarMODWTomp1(x,xdat,len,1,nlevels));
  case BWD:
    printf("\nNot yet implemented!\n");
    return(1);
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarMODWTomp1(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int shift, shift2, lenos, cstart, copyfrom;
  int l2s = 0; // log2(shift)
  int res;

  //*xdat=(real *)malloc(len*2*nlevels*sizeof(real));

  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=skip<<1){
    lenos=len/skip; // we pre-compute this for efficiency
    
    //loop through shifts
    for(shift=0;shift<(skip<<1);shift++){
      shift2 = shift % 2;
      
      cstart = 2*len*l2s + shift*lenos;
      copyfrom = cstart - 2*len - shift2*lenos;

      if(skip==1){
	copyvecskipshiftred(x,1,len,xdat + cstart,shift2);
      }else{
	copyvecskipshiftred(xdat+copyfrom,2,lenos*2,xdat+cstart,shift2);
      }

      // do single level wvt transform on current level
      res = fmpHaar(xdat+cstart,lenos,1,1);

    }
    
    l2s++; //update log2(shift)
  }
  return(0);
}

// this version runs the serial wvt transforms
// but the various shifts are done in parallel
int HaarMODWTomp2(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarMODWTomp2(x,xdat,len,1,nlevels));
  case BWD:
    return(bHaarMODWTomp2(x,xdat,len,1<<(nlevels-1),nlevels));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarMODWTomp2(real* x, real* xdat, uint len, uint skip, uint nlevels){
  //  real* xdat;
  int shift, shift2, lenos, cstart, copyfrom;
  int l2s = 0; // log2(shift)
  int res;

  //*xdat=(real *)malloc(len*2*nlevels*sizeof(real));

  // loop over skip
  for(skip=1;skip<(1<<nlevels);skip=skip<<1){
    lenos=len/skip; // we pre-compute this for efficiency
    
    //loop through shifts
#pragma omp parallel for private (shift,shift2,cstart,copyfrom,res)
    for(shift=0;shift<(skip<<1);shift++){
      shift2 = shift % 2;
      
      cstart = 2*len*l2s + shift*lenos;
      copyfrom = cstart - 2*len - shift2*lenos;

      if(skip==1){
	copyvecskipshiftred(x,1,len,xdat + cstart,shift2);
      }else{
	copyvecskipshiftred(xdat+copyfrom,2,lenos*2,xdat+cstart,shift2);
      }

      // do single level wvt transform on current level
      res = fHaar(xdat+cstart,lenos,1,1);

    }
    
    l2s++; //update log2(shift)
  }
  return(0);
}

int bHaarMODWTomp2(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int shift, shift2, lenos, cstart, copyto;
  int l2s = nlevels-1; // log2(skip)
  int res;
    
  // loop over skip
  for( ; skip>=1; skip=skip>>1){
    // skip is already set before this function is run
    lenos=len/skip; // we pre-compute this for efficiency
      
    //loop through shifts
#pragma omp parallel for private (shift,shift2,cstart,copyto,res)
    for(shift=0;shift<(skip<<1);shift++){
      shift2 = shift % 2;
	
      cstart = 2*len*l2s + shift*lenos;
      copyto = cstart - 2*len - shift2*lenos;
	
      // do single level inv wvt transform on current packet
      res = bHaar(xdat+cstart,lenos,1);
      
      if(shift2==1){
	// we take the average of the two reconstructions
	if(skip>1){
	  // we stay in the xdat vector
	  writeavpackets(xdat+copyto,xdat+cstart-lenos,xdat+cstart,lenos,2);
	}
	else{
	  // we write the final reconstructed x values into the x vector
	  writeavpackets(x,xdat+cstart-lenos,xdat+cstart,lenos,1);
	}
      }
      
    }
    
    l2s--; //update log2(shift) 
  }
  return(0);
}


//#############################################################################
//
// Time-ordered MODWT transform
// - here we don't do copying for in-place transforms
// - instead, we read from latest scaling & write directly to new detail/scaling
// - which are stored seperately & time-ordered
//
//#############################################################################



/*

  data structure used is the following:
  x is our vector

  eg,

  x    = [x0    x1    x2    x3    x4    x5    x6    x7    ]

  which is transformed to vector xdat
  
  xdat = c( s_{1} , d_{1} , s_{2} , d_{2} , s_{3} , d_{3} )

  where c( , ) is R concatenation notation and 

  s_{j} = [sj0 sj1 sj2 sj3 sj4 sj5 sj6 sj7]
  d_{j} = [dj0 dj1 dj2 dj3 dj4 dj5 dj6 dj7]

  these sji and dji are time-ordered coefficients, i.e.
  s_{j}[i] = (1/sqrt(2)) * (s_{j-1}[i] + s_{j-1}[i + skip])
  d_{j}[i] = (1/sqrt(2)) * (s_{j-1}[i] - s_{j-1}[i + skip])

  This version works by reading the "s" coefficieints from the previous level of then writing the transform in the current level.

*/




int fHaarSDout(real* Xin, real* Sout, real* Dout, uint len, uint incri, uint skipi, uint skipo, uint nlevels){
  uint i_i, i_o; // counter inside input, output vector
  if(skipi < (1 << nlevels)){
    for (i_i=0,i_o=0;i_i<len;i_i+=incri,i_o+=skipo){
      Dout[i_o] = (Xin[i_i] - Xin[(i_i+skipi) % len])*invR2;
      Sout[i_o] = (Xin[i_i] + Xin[(i_i+skipi) % len])*invR2;
    }
    return(0);
  }
  else return(1);
}

int bHaarSDout(real* Xout, real* Sin, real* Din, uint len, uint incri, uint skipi, uint skipo, uint nlevels){
  uint i_i, i_o; // counter inside input, output vector
  if(skipi >= 1 ){
    for (i_i=0,i_o=0;i_i<len;i_i+=incri,i_o+=skipo){
      
      Xout[i_o] = (Sin[i_i] + Din[i_i] + Sin[(i_i - skipi) % len] -
		   Din[(i_i - skipi) % len])*hinvR2;
      // this step also does the averaging of coeffs
      // we have the + + - above because each coefficient
      // is reconstructed using each of the filters
    }
    return(0);
  }
  else return(1);
}



int HaarMODWTto(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarMODWTto(x,xdat,len,1,nlevels));
  case BWD:
    return(bHaarMODWTto(x,xdat,len,1<<(nlevels-1),nlevels));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fHaarMODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int lenos, dstart, sstart, readfrom;
  int l; // level
  int res;

  //*xdat=(real *)malloc(len*2*nlevels*sizeof(real));

  // loop over l
  // rather than skip
  for(l=0;l<nlevels;l++){
    if(skip==1){
      sstart = 0;
      dstart = len;
      readfrom = 0;
      res = fHaarSDout(x+readfrom,xdat+sstart,xdat+dstart,len,1,1,1,nlevels);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      readfrom = sstart - 2*len;
      res = fHaarSDout(xdat + readfrom, xdat+sstart, xdat+dstart, len, 1, skip, 1, nlevels);
    }
    skip=skip<<1;
  }
  return(0);
}

int bHaarMODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int lenos, dstart, sstart, copyto;
  int l; // level
  int res;

  // loop over l
  // rather than skip
  for(l=nlevels-1;l>=0;l--){
    if(skip==1){
      sstart = 0;
      dstart = len;
      copyto = 0;
      res = bHaarSDout(x+copyto,xdat+sstart,xdat+dstart,len,1,1,1,nlevels);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      copyto = sstart - 2*len;
      res = bHaarSDout(xdat + copyto, xdat+sstart, xdat+dstart, len, 1, skip, 1, nlevels);
    }
    skip=skip>>1;
  }
  return(0);
}



int fHaarSDoutomp(real* Xin, real* Sout, real* Dout, uint len, uint incri, uint skipi, uint skipo, uint nlevels){
  uint i_i, i_o, i; // counter inside input, output vector
  if(skipi < (1 << nlevels)){
#pragma omp parallel for private (i,i_i,i_o)
    for (i=0;i<len/incri;i++){
      i_i = i*incri;
      i_o = i*skipo;
      Dout[i_o] = (Xin[i_i] - Xin[(i_i+skipi) % len])*invR2;
      Sout[i_o] = (Xin[i_i] + Xin[(i_i+skipi) % len])*invR2;
    }
    return(0);
  }
  else return(1);
}

int bHaarSDoutomp(real* Xout, real* Sin, real* Din, uint len, uint incri, uint skipi, uint skipo, uint nlevels){
  uint i_i, i_o, i; // counter inside input, output vector
  if(skipi >= 1 ){
#pragma omp parallel for private (i,i_i,i_o)
    for (i=0;i<len/incri;i++){
      i_i = i*incri;
      i_o = i*skipo;
      
      Xout[i_o] = (Sin[i_i] + Din[i_i] + Sin[(i_i - skipi) % len] -
		   Din[(i_i - skipi) % len])*hinvR2;
    }
    return(0);
  }
  else return(1);
}


int HaarMODWTtomp(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=2;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fHaarMODWTtomp(x,xdat,len,1,nlevels));
  case BWD:
    return(bHaarMODWTtomp(x,xdat,len,1<<(nlevels-1),nlevels));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fHaarMODWTtomp(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int lenos, dstart, sstart, readfrom;
  int l; // level
  int res;

  //*xdat=(real *)malloc(len*2*nlevels*sizeof(real));

  // loop over l
  // rather than skip
  for(l=0;l<nlevels;l++){
    if(skip==1){
      sstart = 0;
      dstart = len;
      readfrom = 0;
      res = fHaarSDoutomp(x+readfrom,xdat+sstart,xdat+dstart,len,1,1,1,nlevels);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      readfrom = sstart - 2*len;
      res = fHaarSDoutomp(xdat + readfrom, xdat+sstart, xdat+dstart, len, 1, skip, 1, nlevels);
    }
    skip=skip<<1;
  }
  return(0);
}


int bHaarMODWTtomp(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int lenos, dstart, sstart, copyto;
  int l; // level
  int res;

  // loop over l
  // rather than skip
  for(l=nlevels-1;l>=0;l--){
    if(skip==1){
      sstart = 0;
      dstart = len;
      copyto = 0;
      res = bHaarSDoutomp(x+copyto,xdat+sstart,xdat+dstart,len,1,1,1,nlevels);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      copyto = sstart - 2*len;
      res = bHaarSDoutomp(xdat + copyto, xdat+sstart, xdat+dstart, len, 1, skip, 1, nlevels);
    }
    skip=skip>>1;
  }
  return(0);
}

// an idea. But we're doing our reconstruction in-place in x_dat
// int bHaar1lPO(real* xr, real* xp0, real* xp1 , uint len){
//   uint len2 = len << 1; // 2 * len
//   // this is the length of the packet at level j-1
//   // whilst the length of the packet at level j is len
//   uint i, i2=0;
//   for(i=0;i<len;i++){
//     xr[i2] = ( (xp0[i] + xp0[i+1]) + (xp1[i+1]

//     i2+=2;
//   }
// }
