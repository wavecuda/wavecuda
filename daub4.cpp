#include "daub4.h"
#include "daub4coeffs.h"
#include "utils.h"

int Daub4(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fDaub4(x,len,1,nlevels));
  case BWD:
    return(bDaub4(x,len,1<<(nlevels-1))); // correction: used to be len/4
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fDaub4(real* x, uint len, uint skip, uint nlevels){
  real tmp, x0, xskip;
  uint i;
  // static uint opsf = 0;
  // static uint loops = 0;
  // uint i, i2, i3;
  
  if(skip < (1 << nlevels)){
    
    //store first 2 entries of vector
    x0=x[0]; xskip=x[skip];

    //do loop from i=0 -> i < len-3*skip : does transform to store in i=0 -> i=len-2*skip
    for(i=0;i<(len - 3*skip);i+=2*skip){
      // printf("\nlen=%i,skip=%i,i=%i,i+skip=%i,i+2skip=%i,i+3skip=%i",len,skip,i,i+skip,i+2*skip,i+3*skip); 
      
      tmp = x[i]*C3 - x[i+skip]*C2 + x[i+2*skip]*C1 - x[i+3*skip]*C0; //detail
      x[i] = x[i]*C0 + x[i+skip]*C1 + x[i+2*skip]*C2 + x[i+3*skip]*C3; //scaling
      x[i+skip] = tmp;
      // opsf+=14;
      // loops+=1;
    }

    //do last cycle of loop manually
    tmp =  x[len-2*skip]*C3 - x[len-skip]*C2 + x0*C1 - xskip*C0;
    x[len-2*skip] = x[len-2*skip]*C0 + x[len-skip]*C1 + x0*C2 + xskip*C3;
    x[len-skip] = tmp;
    // opsf+=14;
    // loops+=1;
    // printvec(x,len);
    
    return(fDaub4(x,len,skip<<1,nlevels));
  }  
  // printf("\nOpsf = %u",opsf);
  // printf("\nLoops = %u",loops);
  return(0);
}


int bDaub4(real* x, uint len, uint skip){
  real tmp,x0,xskip;
  // uint i, i2, i3;
  uint i;
  
  if(skip > 0){
    
    //do first 2 trans seperately, as elts 0 & skip calculated from len -2*skip, len - skip, 0 & 1
    x0 = x[len-2*skip]*C2 + x[len-skip]*C1 + x[0]*C0 + x[skip]*C3;
    xskip = x[len-2*skip]*C3 - x[len-skip]*C0 + x[0]*C1 - x[skip]*C2;
    
    //loop backward stransform i=2*skip -> i < len-3*skip
    //loop will happen at least once, as skip<=len/4
    for(i=len-2*skip;i>=2*skip;i-=2*skip){

      tmp = x[i-2*skip]*C3 - x[i-skip]*C0 + x[i]*C1 - x[i+skip]*C2;
      x[i] = x[i-2*skip]*C2 + x[i-skip]*C1 + x[i]*C0 + x[i+skip]*C3;
      x[i+skip] = tmp;

    }
    
    x[0]=x0;
    x[skip]=xskip;

    // printvec(x,len);
    
    return(bDaub4(x,len,skip>>1));
  }
  return(0);
}

/******************************************************
Lifted Daubechies 4 code
******************************************************/

int lDaub4(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(flDaub4(x,len,1,nlevels));
  case BWD:
    return(blDaub4(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int flDaub4(real* x, uint len, uint skip, uint nlevels){
  real xfirst;
  uint i;
  // with periodic boundary conditions
  
  if(skip < (1 << nlevels)){

    for(i=0;i<len;i+=2*skip){
      x[i] = x[i] + x[i+skip]*Cl0;
      // s1[l] = x[2l] + sqrt(3)*x[2l+1]
    }

    //do first cycle manually
    x[skip] = x[skip] - x[0]*Cl1 - x[len-2*skip]*Cl2;
    for(i=2*skip;i<len;i+=2*skip){
      x[i+skip] = x[i+skip] - x[i]*Cl1 - x[i-2*skip]*Cl2;
      // d1[l] = x[2l+1] - s1[l]*sqrt(3)/4 - s1[l-1]*(sqrt(3)-2)/4
    }
    
    for(i=0;i<(len-2*skip);i+=2*skip){
      x[i] = x[i] - x[i+3*skip];
      // s2[l] = s1[l] - d1[l+1]
    }
    x[len-2*skip] = x[len-2*skip] - x[skip];
    //do last cycle manually

    for(i=0;i<len;i+=2*skip){
      x[i] = x[i]*Cl3;
      // s[l] = s2[l]*(sqrt(3)-1)/sqrt(2)
      x[i+skip] = x[i+skip]*Cl4;
      // d[l] = d1[l]*(sqrt(3)+1)/sqrt(2)
    }

    return(flDaub4(x,len,skip<<1,nlevels));
  }      
  return(0);
}

int blDaub4(real* x, uint len, uint skip){
  uint i;
  if(skip > 0){
    
    for(i=0;i<len;i+=2*skip){
      x[i] = x[i]*Cl4;
      // s2[l] = s[l]*(sqrt(3)+1)/sqrt(2)
      x[i+skip] = x[i+skip]*Cl3;
      // d1[l] = d[l]*(sqrt(3)-1)/sqrt(2)  
    }

    for(i=0;i<(len-2*skip);i+=2*skip){
      x[i] = x[i] + x[i+3*skip];
      // s1[l] = s2[l] + d1[l+1]
    }
    x[len-2*skip] = x[len-2*skip] + x[skip];
    //do last cycle manually
    
    //do first cycle manually
    x[skip] = x[skip] + x[0]*Cl1 + x[len-2*skip]*Cl2;
    for(i=2*skip;i<len;i+=2*skip){
      x[i+skip] = x[i+skip] + x[i]*Cl1 + x[i-2*skip]*Cl2;
      // x[2l+1] = d1[l] + s1[l]*sqrt(3)/4 + s1[l-1]*(sqrt(3)-2)/4
    }

    for(i=0;i<len;i+=2*skip){
      x[i] = x[i] - x[i+skip]*Cl0;
      // x[2l] = s1[l] - sqrt(3)*d1[l]
    }

     // printvec(x,len);

    return(blDaub4(x,len,skip>>1));
  }
  return(0);
}

/******************************************************
Lifted Daubechies 4 code - with Omp
******************************************************/

int lompDaub4(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(flompDaub4(x,len,1,nlevels));
  case BWD:
    return(blompDaub4(x,len,len/4));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int flompDaub4(real* x, uint len, uint skip, uint nlevels){
  real xfirst;
  uint i;
  // with periodic boundary conditions
  
  if(skip < (1 << nlevels)){
#pragma omp parallel for private (i)        
    for(i=0;i<len;i+=2*skip){
      x[i] = x[i] + x[i+skip]*Cl0;
      // s1[l] = x[2l] + sqrt(3)*x[2l+1]
    }
    //do first cycle manually
    x[skip] = x[skip] - x[0]*Cl1 - x[len-2*skip]*Cl2;
#pragma omp parallel for private (i)        
    for(i=2*skip;i<len;i+=2*skip){
      x[i+skip] = x[i+skip] - x[i]*Cl1 - x[i-2*skip]*Cl2;
      // d1[l] = x[2l+1] - s1[l]*sqrt(3)/4 - s1[l-1]*(sqrt(3)-2)/4
    }
#pragma omp parallel for private (i)        
    for(i=0;i<(len-2*skip);i+=2*skip){
      x[i] = x[i] - x[i+3*skip];
      // s2[l] = s1[l] - d1[l+1]
    }
    x[len-2*skip] = x[len-2*skip] - x[skip];
    //do last cycle manually
#pragma omp parallel for private (i)    
    for(i=0;i<len;i+=2*skip){
      x[i] = x[i]*Cl3;
      // s[l] = s2[l]*(sqrt(3)-1)/sqrt(2)
      x[i+skip] = x[i+skip]*Cl4;
      // d[l] = d1[l]*(sqrt(3)+1)/sqrt(2)
    }

    // xfirst=x[skip];
    // for(i=0;i<len-2*skip;i+=2*skip){
    //   x[i+skip] = -x[i+3*skip];
    // }
    // x[len-skip]=-xfirst;

    //printvec(x,len);

    return(flompDaub4(x,len,skip<<1,nlevels));
  }      
  return(0);
}

int blompDaub4(real* x, uint len, uint skip){
  uint i;
  if(skip > 0){
#pragma omp parallel for private (i)    
    for(i=0;i<len;i+=2*skip){
      x[i] = x[i]*Cl4;
      // s2[l] = s[l]*(sqrt(3)+1)/sqrt(2)
      x[i+skip] = x[i+skip]*Cl3;
      // d1[l] = d[l]*(sqrt(3)-1)/sqrt(2)  
    }
#pragma omp parallel for private (i)    
    for(i=0;i<(len-2*skip);i+=2*skip){
      x[i] = x[i] + x[i+3*skip];
      // s1[l] = s2[l] + d1[l+1]
    }
    x[len-2*skip] = x[len-2*skip] + x[skip];
    //do last cycle manually    
    //do first cycle manually
    x[skip] = x[skip] + x[0]*Cl1 + x[len-2*skip]*Cl2;
#pragma omp parallel for private (i)    
    for(i=2*skip;i<len;i+=2*skip){
      x[i+skip] = x[i+skip] + x[i]*Cl1 + x[i-2*skip]*Cl2;
      // x[2l+1] = d1[l] + s1[l]*sqrt(3)/4 + s1[l-1]*(sqrt(3)-2)/4
    }
#pragma omp parallel for private (i)    
    for(i=0;i<len;i+=2*skip){
      x[i] = x[i] - x[i+skip]*Cl0;
      // x[2l] = s1[l] - sqrt(3)*d1[l]
    }

    // printvec(x,len);

    return(blompDaub4(x,len,skip>>1));
  }
  return(0);
}


/*****************************************************************
Alternative lifting algo with more flops
******************************************************************/

int l2Daub4(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case 1:
    return(fl2Daub4(x,len,1,nlevels));
  case 0:
    return(bl2Daub4(x,len,len/4));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fl2Daub4(real* x, uint len, uint skip, uint nlevels){
  uint i;
  // with periodic boundary conditions
  
  if(skip < (1 << nlevels)){

    for(i=0;i<len-2*skip;i+=2*skip){
      x[i+skip] = x[i+skip] - x[i+2*skip]*Cl20;
      // d1[l] = x[2l+1] - 1/sqrt(3) *x[2l+2]
    }
    x[len-skip] = x[len-skip] - x[0]*Cl20;
    //do last cycle manually

    //do first cycle manually
    x[0] = x[0] + x[skip]*Cl21 + x[len-skip]*Cl22;
    for(i=2*skip;i<len;i+=2*skip){
      x[i] = x[i] + x[i+skip]*Cl21 + x[i-skip]*Cl22;
      // s1[l] = x[2l] + d1[l]*(6-3*sqrt(3))/4 + d1[l-1]*sqrt(3)/4
    }

    for(i=0;i<len;i+=2*skip){
      x[i+skip] = x[i+skip] - x[i]*Cl23;
      // d2[l] = d1[l] - 1/3 *s[l]
    }

    for(i=0;i<len;i+=2*skip){
      x[i] = x[i]*Cl24;
      // s[l] = s1[l]*(3+sqrt(3))/(3*sqrt(2))
      x[i+skip] = x[i+skip]*Cl25;
      // d[l] = d2[l]*(3-sqrt(3))/(sqrt(2))
    }

    // printvec(x,len);

    return(fl2Daub4(x,len,skip<<1,nlevels));
  }      
  return(0);
}


int bl2Daub4(real* x, uint len, uint skip){
  uint i;
  if(skip > 0){

    for(i=0;i<len;i+=2*skip){
      x[i] = x[i]*Cl25;
      // s1[l] = s[l]*(3-sqrt(3))/(sqrt(2))
      x[i+skip] = x[i+skip]*Cl24;
      // d2[l] = d[l]*(3+sqrt(3))/(3*sqrt(2))
    }

    for(i=0;i<len;i+=2*skip){
      x[i+skip] = x[i+skip] + x[i]*Cl23;
      // d1[l] = d2[l] + 1/3 *s[l]
    }

    //do first cycle manually
    x[0] = x[0] - x[skip]*Cl21 - x[len-skip]*Cl22;
    for(i=2*skip;i<len;i+=2*skip){
      x[i] = x[i] - x[i+skip]*Cl21 - x[i-skip]*Cl22;
      // x[2l] = s1[l] - d1[l]*(6-3*sqrt(3))/4 - d1[l-1]*sqrt(3)/4
    }

    for(i=0;i<len-2*skip;i+=2*skip){
      x[i+skip] = x[i+skip] + x[i+2*skip]*Cl20;
      // x[2l+1] = d1[l] + 1/sqrt(3) *x[2l+2]
    }
    x[len-skip] = x[len-skip] + x[0]*Cl20;
    //do last cycle manually

    //printvec(x,len);

    return(bl2Daub4(x,len,skip>>1));
  }
  return(0);
}



//#############################################################################
//
// Packet-ordered Daub4 MODWT transform
//
//#############################################################################


int Daub4MODWTpo(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fDaub4MODWTpo(x,xdat,len,1,nlevels));
  case BWD:
    return(bDaub4MODWTpo(x,xdat,len,1<<(nlevels-1),nlevels));
    //    printf("\nNot yet implemented!\n");
    //    return(1);
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fDaub4MODWTpo(real* x, real* xdat, uint len, uint skip, uint nlevels){
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
      res = fDaub4(xdat+cstart,lenos,1,1);

    }
    
    l2s++; //update log2(shift)
  }
  return(0);
}

int bDaub4MODWTpo(real* x, real* xdat, uint len, uint skip, uint nlevels){
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
      res = bDaub4(xdat+cstart,lenos,1);
      
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
// Time-ordered Daub4 MODWT transform
//
//#############################################################################



int Daub4MODWTto(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(fDaub4MODWTto(x,xdat,len,1,nlevels));
  case BWD:
    return(bDaub4MODWTto(x,xdat,len,1<<(nlevels-1),nlevels));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fDaub4MODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int lenos, dstart, sstart, readfrom;
  int l; // level
  int res;

  // loop over l
  // rather than skip
  for(l=0;l<nlevels;l++){
    if(skip==1){
      sstart = 0;
      dstart = len;
      readfrom = 0;
      res = fDaub4SDout(x+readfrom,xdat+sstart,xdat+dstart,len,1,1,1,nlevels);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      readfrom = sstart - 2*len;
      res = fDaub4SDout(xdat + readfrom, xdat+sstart, xdat+dstart, len, 1, skip, 1, nlevels);
    }
    skip=skip<<1;
  }
  return(0);
}

int bDaub4MODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels){
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
      res = bDaub4SDout(x+copyto,xdat+sstart,xdat+dstart,len,1,1,1,nlevels);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      copyto = sstart - 2*len;
      res = bDaub4SDout(xdat + copyto, xdat+sstart, xdat+dstart, len, 1, skip, 1, nlevels);
    }
    skip=skip>>1;
  }
  return(0);
}


int fDaub4SDout(real* Xin, real* Sout, real* Dout, uint len, uint incri, uint skipi, uint skipo, uint nlevels){
  uint i_i, i_o; // counter inside input, output vector
  if(skipi < (1 << nlevels)){
    for (i_i=0,i_o=0;i_i<len;i_i+=incri,i_o+=skipo){
      Dout[i_o] = Xin[i_i]*C3 - Xin[(i_i+skipi) % len]*C2 + Xin[(i_i+2*skipi) % len]*C1 - Xin[(i_i+3*skipi) % len]*C0;
      Sout[i_o] = Xin[i_i]*C0 + Xin[(i_i+skipi) % len]*C1 + Xin[(i_i+2*skipi) % len]*C2 + Xin[(i_i+3*skipi) % len]*C3;
    }
    return(0);
  }
  else return(1);
}

int bDaub4SDout(real* Xout, real* Sin, real* Din, uint len, uint incri, uint skipi, uint skipo, uint nlevels){
  uint i_i, i_o; // counter inside input, output vector
  if(skipi >= 1 ){
    for (i_i=0,i_o=0;i_i<len;i_i+=incri,i_o+=skipo){
      
       Xout[i_o] = 0.5*(
			Sin[(i_i - 2*skipi) % len]*C2 + Din[(i_i - 2*skipi) % len]*C1 + Sin[i_i]*C0 + Din[i_i]*C3
			+
			Sin[(i_i - 3*skipi) % len]*C3 - Din[(i_i - 3*skipi) % len]*C0 + Sin[(i_i - skipi) % len]*C1 - Din[(i_i - skipi) % len]*C2
			);
       
       // this step also does the averaging of coeffs
    }
    return(0);
  }
  else return(1);
}

//#############################################################################
//
// Packet-ordered Daub4 MODWT transform
//
//#############################################################################


int lDaub4MODWTpo(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(flDaub4MODWTpo(x,xdat,len,1,nlevels));
  case BWD:
    return(blDaub4MODWTpo(x,xdat,len,1<<(nlevels-1),nlevels));
    //    printf("\nNot yet implemented!\n");
    //    return(1);
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int flDaub4MODWTpo(real* x, real* xdat, uint len, uint skip, uint nlevels){
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
      res = flDaub4(xdat+cstart,lenos,1,1);

    }
    
    l2s++; //update log2(shift)
  }
  return(0);
}

int blDaub4MODWTpo(real* x, real* xdat, uint len, uint skip, uint nlevels){
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
      res = blDaub4(xdat+cstart,lenos,1);
      
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
// Time-ordered Daub4 MODWT transform
// - lifting version
//
//#############################################################################

int lDaub4MODWTto(real* x, real* xdat, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=4;
  nlevels = check_len_levels(len,nlevels,filterlength);
  switch(sense){
  case FWD:
    return(flDaub4MODWTto(x,xdat,len,1,nlevels));
  case BWD:
    return(blDaub4MODWTto(x,xdat,len,1<<(nlevels-1),nlevels));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int flDaub4MODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels){
  int lenos, dstart, sstart, readfrom;
  int l; // level
  int res;

  // loop over l
  // rather than skip
  for(l=0;l<nlevels;l++){
    if(skip==1){
      sstart = 0;
      dstart = len;
      readfrom = 0;
      res = flDaub4SDout(x+readfrom,xdat+sstart,xdat+dstart,len,1,1,1,nlevels);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      readfrom = sstart - 2*len;
      res = flDaub4SDout(xdat + readfrom, xdat+sstart, xdat+dstart, len, 1, skip, 1, nlevels);
    }
    skip=skip<<1;
  }
  return(0);
}

int blDaub4MODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels){
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
      res = blDaub4SDout(x+copyto,xdat+sstart,xdat+dstart,len,1,1,1,nlevels);
    }
    else{
      sstart = len * 2 * l;
      dstart = len * (2 * l + 1);
      copyto = sstart - 2*len;
      res = blDaub4SDout(xdat + copyto, xdat+sstart, xdat+dstart, len, 1, skip, 1, nlevels);
    }
    skip=skip>>1;
  }
  return(0);
}

int flDaub4SDout(real* Xin, real* Sout, real* Dout, uint len, uint incri, uint skipi, uint skipo, uint nlevels){
  uint i_i, i_o; // counter inside input, output vector
  if(skipi < (1 << nlevels)){
    for (i_i=0,i_o=0;i_i<len;i_i+=incri,i_o+=skipo){
      Sout[i_o] = Xin[i_i] + Xin[(i_i+skipi) % len]*Cl0;
    }
    
    for (i_i=0,i_o=0;i_i<len;i_i+=incri,i_o+=skipo){
      Dout[i_o] = Xin[(i_i+skipi) % len] - Sout[i_o]*Cl1 - Sout[(i_o-2*skipi) % len]*Cl2;
    }
    
    for (i_o=0;i_o<len;i_o+=skipo){
      Sout[i_o] = Sout[i_o] - Dout[(i_o + 2*skipi) % len];
    }
    
    for (i_o=0;i_o<len;i_o+=skipo){
      Sout[i_o] = Sout[i_o]*Cl3;
      Dout[i_o] = Dout[i_o]*Cl4;
    }
    
    return(0);
  }
  else return(1);
}



int blDaub4SDout(real* Xout, real* Sin, real* Din, uint len, uint incri, uint skipi, uint skipo, uint nlevels){
  uint i_i, i_o; // counter inside input, output vector
  if(skipi >= 1 ){

    // ## reconstruction lifting steps ---->---->----> ##

    for (i_i=0;i_i<len;i_i+=incri){
      Sin[i_i] = Sin[i_i]*Cl4;
      Din[i_i] = Din[i_i]*Cl3;
    }

        for (i_i=0;i_i<len;i_i+=incri){
      Sin[i_i] = Sin[i_i] + Din[(i_i + 2*skipi) % len];
    }

    for (i_i=0;i_i<len;i_i+=incri){
      Din[i_i] = Din[(i_i) % len] + Sin[i_i]*Cl1 + Sin[(i_i-2*skipi) % len]*Cl2;
    }

    for (i_i=0;i_i<len;i_i+=incri){
      Sin[i_i] = Sin[i_i] - Din[i_i]*Cl0;
    }

    // ## <----<----<---- reconstruction lifting steps ##
    
    for (i_i=0,i_o=0;i_i<len;i_i+=incri,i_o+=skipo){
      Xout[i_o] = 0.5*(Sin[i_i] + Din[(i_i - skipi) % len]);
    }
    
    // then we average the reconstructions, one scaling & detail
    
    return(0);
  }
  else return(1);
}
