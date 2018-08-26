#include "c6.h"
#include "c6coeffs.h"
#include "utils.h"

#define DEBUGL 0
// simple little macro to debug the lifted versions' coefficients
// where we shift & multiply by -1 to match the standard implementation


// ##############################################################
// First, the plain, boring, slow C6 transform
// puts x[i] = C * x[i..i+5*skip]
// ##############################################################

int C6(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=6;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels==0) return(1);
  switch(sense){
  case FWD:
    return(fC6(x,len,1,nlevels));
  case BWD:
    return(bC6(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int fC6(real* x, uint len, uint skip, uint nlevels){
  real tmp;
  uint i;
  // static uint opsf = 0;
  // static uint loops = 0;
  // uint i, i2, i3;
  
  if(skip < (1 << nlevels)){
    
    //store first 4 entries of vector
    real x0=x[0], xskip=x[skip], x2skip=x[2*skip], x3skip=x[3*skip];

    //do loop from i=0 -> i < len-3*skip : does transform to store in i=0 -> i=len-2*skip
    for(i=0;i<=(len - 6*skip);i+=(skip<<1)){
      // printf("\nlen=%i,skip=%i,i=%i,i+skip=%i,i+2skip=%i,i+3skip=%i",len,skip,i,i+skip,i+2*skip,i+3*skip); 
      
      tmp = x[i]*C5 - x[i+skip]*C4 + x[i+2*skip]*C3 - x[i+3*skip]*C2 + x[i+4*skip]*C1 - x[i+5*skip]*C0; //detail
      x[i] = x[i]*C0 + x[i+skip]*C1 + x[i+2*skip]*C2 + x[i+3*skip]*C3 + x[i+4*skip]*C4 + x[i+5*skip]*C5; //scaling
      x[i+skip] = tmp;
      // opsf+=11;
      // loops+=1;
    }

    // do the last 2 iterations manually!

    i=len-4*skip; // should already be the case, but just making sure!
    tmp = x[i]*C5 - x[i+skip]*C4 + x[i+2*skip]*C3 - x[i+3*skip]*C2 + x0*C1 - xskip*C0; //detail
    x[i] = x[i]*C0 + x[i+skip]*C1 + x[i+2*skip]*C2 + x[i+3*skip]*C3 + x0*C4 + xskip*C5; //scaling
    x[i+skip] = tmp;
    
    i+=(skip<<1); //len-2*skip;
    tmp = x[i]*C5 - x[i+skip]*C4 + x0*C3 - xskip*C2 + x2skip*C1 - x3skip*C0; //detail
    x[i] = x[i]*C0 + x[i+skip]*C1 + x0*C2 + xskip*C3 + x2skip*C4 + x3skip*C5; //scaling
    x[i+skip] = tmp;

    // opsf+=;
    // loops+=1;
    // printvec(x,len);
    
    return(fC6(x,len,skip<<1,nlevels));
  }  
  // printf("\nOpsf = %u",opsf);
  // printf("\nLoops = %u",loops);
  return(0);
}


int bC6(real* x, uint len, uint skip){
  real tmp;
  // uint i, i2, i3;
  uint i;
  
  if(skip > 0){
        
    //store 4 entries of vector
    real xlskip=x[len-skip], xl2skip=x[len-2*skip], xl3skip=x[len-3*skip], xl4skip=x[len-4*skip];

    for(i=len-2*skip;i>=(4*skip);i-=(skip<<1)){

      tmp = x[i-4*skip]*C5 - x[i-3*skip]*C0 + x[i-2*skip]*C3 - x[i-skip]*C2 + x[i]*C1 - x[i+skip]*C4; //detail
      x[i] = x[i-4*skip]*C4 + x[i-3*skip]*C1 + x[i-2*skip]*C2 + x[i-skip]*C3 + x[i]*C0 + x[i+skip]*C5; //scaling
      x[i+skip] = tmp;
    }
    
    i=2*skip; // should already be the case, but just making sure!
    tmp = xl2skip*C5 - xlskip*C0 + x[i-2*skip]*C3 - x[i-skip]*C2 + x[i]*C1 - x[i+skip]*C4; //detail
    x[i] = xl2skip*C4 + xlskip*C1 + x[i-2*skip]*C2 + x[i-skip]*C3 + x[i]*C0 + x[i+skip]*C5; //scaling
    x[i+skip] = tmp;
    
    i-=(skip<<1); //0;
    tmp = xl4skip*C5 - xl3skip*C0 + xl2skip*C3 - xlskip*C2 + x[i]*C1 - x[i+skip]*C4; //detail
    x[i] = xl4skip*C4 + xl3skip*C1 + xl2skip*C2 + xlskip*C3 + x[i]*C0 + x[i+skip]*C5; //scaling
    x[i+skip] = tmp;
    
    // printvec(x,len);
    
    return(bC6(x,len,skip>>1));
  }
  return(0);
}



// ##############################################################
// Now, the lifting C6 version, which should be faster!
// but has different time alignment
// puts x[i] = C * x[i-2*skip..i+3*skip]
// and x[i+1] = (-1) * rev(C) * x[i-2*skip...i+3*skip]
// ##############################################################


int lC6(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=6;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels==0) return(1);
  switch(sense){
  case FWD:
    return(flC6(x,len,1,nlevels));
  case BWD:
    return(blC6(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}



int flC6(real*x, uint len, uint skip, uint nlevels){
  uint i;
  // with periodic boundary conditions

  if(skip < (1 << nlevels)){
    
    // first lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] + CL0*x[i];
      // d1[l] = x[2l+1] + q11*x[2l]
    }
    // printf("\n## cpp lifting -- 1st loop done ##");
    // printvec(x,len);

    // second lifting loop
    for(i=0;i<len-2*skip;i+=(skip<<1)){
      x[i] = x[i] + CL1*x[i+3*skip];
      // s1[l] = x[2l] + q21*x[2l+3]
    }
    x[len-2*skip] = x[len-2*skip] + CL1*x[skip];
    // printf("\n## cpp lifting -- 2nd loop done ##");
    // printvec(x,len);

    // third lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] + CL2*x[i];
      // d2[l] = x[2l+1] + q31*x[2l]
    }
    // printf("\n## cpp lifting -- 3rd loop done ##");
    // printvec(x,len);
    
    // fourth lifting loop
    x[0] = x[0] + CL3*x[len-skip] + CL4*x[skip];
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] + CL3*x[i-skip] + CL4*x[i+skip];
      // s2[l] = x[2l] + q41*x[2l-1] + q42*x[2l+1]
    }
    // printf("\n## cpp lifting -- 4th loop done ##");
    // printvec(x,len);

    // fifth lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] + CL5*x[i];
      // d3[l] = x[2l+1] + s*K^2*x[2l]
    }
    // printf("\n## cpp lifting -- 5th loop done ##");
    // printvec(x,len);

    // sixth lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      x[i] = CL6*x[i];
      //s3[l] = (K)*s2[l]      
      x[i+skip] = CL7*x[i+skip];
      // d4[l] = (1/K)*d3[l]
    }
    // printf("\n## cpp lifting -- 6th loop done ##");
    // printvec(x,len);
    
    if(DEBUGL){
      shift_wvt_vector(x,len,skip,-1,-1);
      ax_wvt_vector(x,len,skip,-1,1);
    }

    return(flC6(x,len,skip<<1,nlevels));
  }

  return(0);
}


int blC6(real* x, uint len, uint skip){
  uint i;
  
  if(skip > 0){
    
    if(DEBUGL){
      shift_wvt_vector(x,len,skip,1,1);
      ax_wvt_vector(x,len,skip,-1,1);
    }

    // sixth lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      x[i] = CL7*x[i];
      //s2[l] = (1/K)*s3[l]      
      x[i+skip] = CL6*x[i+skip];
      // d3[l] = (K)*d4[l]
    }

    // fifth lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] - CL5*x[i];
      // d2[l] = x[2l+1] - s*K^2*x[2l]
    }
    // printf("\n## cpp lifting -- 5th loop done ##");
    // printvec(x,len);
    
    // fourth lifting loop
    x[0] = x[0] - CL3*x[len-skip] - CL4*x[skip];
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] - CL3*x[i-skip] - CL4*x[i+skip];
      // s1[l] = x[2l] - q41*x[2l-1] - q42*x[2l+1]
    }
    // printf("\n## cpp lifting -- 4th loop done ##");
    // printvec(x,len);

    // third lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] - CL2*x[i];
      // d1[l] = x[2l+1] - q31*x[2l]
    }
    // printf("\n## cpp lifting -- 3rd loop done ##");
    // printvec(x,len);

    // second lifting loop
    for(i=0;i<len-2*skip;i+=(skip<<1)){
      x[i] = x[i] - CL1*x[i+3*skip];
      // s0[l] = x[2l] - q21*x[2l+3]
    }
    x[len-2*skip] = x[len-2*skip] - CL1*x[skip];
    // printf("\n## cpp lifting -- 2nd loop done ##");
    // printvec(x,len);

    // first lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] - CL0*x[i];
      // d0[l] = x[2l+1] - q11*x[2l]
    }
    // printf("\n## cpp lifting -- 1st loop done ##");
    // printvec(x,len);

    return(blC6(x,len,skip>>1));
  }
  
  return(0);
}



// ##############################################################
// Now, the lifting C6 version, OMP versions!
// with time alignment
// puts x[i] = C * x[i-2*skip..i+3*skip]
// and x[i+1] = (-1) * rev(C) * x[i-2*skip...i+3*skip]
// ##############################################################


int lompC6(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=6;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels==0) return(1);
  switch(sense){
  case FWD:
    return(flompC6(x,len,1,nlevels));
  case BWD:
    return(blompC6(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}



int flompC6(real*x, uint len, uint skip, uint nlevels){
  uint i;
  // with periodic boundary conditions

  if(skip < (1 << nlevels)){
    
    // first lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] + CL0*x[i];
      // d1[l] = x[2l+1] + q11*x[2l]
    }
    // printf("\n## cpp lifting -- 1st loop done ##");
    // printvec(x,len);

    // second lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len-2*skip;i+=(skip<<1)){
      x[i] = x[i] + CL1*x[i+3*skip];
      // s1[l] = x[2l] + q21*x[2l+3]
    }
    x[len-2*skip] = x[len-2*skip] + CL1*x[skip];
    // printf("\n## cpp lifting -- 2nd loop done ##");
    // printvec(x,len);

    // third lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] + CL2*x[i];
      // d2[l] = x[2l+1] + q31*x[2l]
    }
    // printf("\n## cpp lifting -- 3rd loop done ##");
    // printvec(x,len);
    
    // fourth lifting loop
    x[0] = x[0] + CL3*x[len-skip] + CL4*x[skip];
#pragma omp parallel for private (i)
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] + CL3*x[i-skip] + CL4*x[i+skip];
      // s2[l] = x[2l] + q41*x[2l-1] + q42*x[2l+1]
    }
    // printf("\n## cpp lifting -- 4th loop done ##");
    // printvec(x,len);

    // fifth lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] + CL5*x[i];
      // d3[l] = x[2l+1] + s*K^2*x[2l]
    }
    // printf("\n## cpp lifting -- 5th loop done ##");
    // printvec(x,len);

    // sixth lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len;i+=(skip<<1)){
      x[i] = CL6*x[i];
      //s3[l] = (K)*s2[l]      
      x[i+skip] = CL7*x[i+skip];
      // d4[l] = (1/K)*d3[l]
    }
    // printf("\n## cpp lifting -- 6th loop done ##");
    // printvec(x,len);
    
    return(flompC6(x,len,skip<<1,nlevels));
  }

  return(0);
}


int blompC6(real* x, uint len, uint skip){
  uint i;
  
  if(skip > 0){
    
    // sixth lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len;i+=(skip<<1)){
      x[i] = CL7*x[i];
      //s2[l] = (1/K)*s3[l]      
      x[i+skip] = CL6*x[i+skip];
      // d3[l] = (K)*d4[l]
    }

    // fifth lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] - CL5*x[i];
      // d2[l] = x[2l+1] - s*K^2*x[2l]
    }
    // printf("\n## cpp lifting -- 5th loop done ##");
    // printvec(x,len);
    
    // fourth lifting loop
    x[0] = x[0] - CL3*x[len-skip] - CL4*x[skip];
#pragma omp parallel for private (i)
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] - CL3*x[i-skip] - CL4*x[i+skip];
      // s1[l] = x[2l] - q41*x[2l-1] - q42*x[2l+1]
    }
    // printf("\n## cpp lifting -- 4th loop done ##");
    // printvec(x,len);

    // third lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] - CL2*x[i];
      // d1[l] = x[2l+1] - q31*x[2l]
    }
    // printf("\n## cpp lifting -- 3rd loop done ##");
    // printvec(x,len);

    // second lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len-2*skip;i+=(skip<<1)){
      x[i] = x[i] - CL1*x[i+3*skip];
      // s0[l] = x[2l] - q21*x[2l+3]
    }
    x[len-2*skip] = x[len-2*skip] - CL1*x[skip];
    // printf("\n## cpp lifting -- 2nd loop done ##");
    // printvec(x,len);

    // first lifting loop
#pragma omp parallel for private (i)
    for(i=0;i<len;i+=(skip<<1)){
      x[i+skip] = x[i+skip] - CL0*x[i];
      // d0[l] = x[2l+1] - q11*x[2l]
    }
    // printf("\n## cpp lifting -- 1st loop done ##");
    // printvec(x,len);

    return(blompC6(x,len,skip>>1));
  }
  
  return(0);
}
