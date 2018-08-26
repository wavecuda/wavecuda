#include "la8.h"
#include "la8coeffs.h"
#include "utils.h"

// ##############################################################
// First, the plain, boring, slow LA8 transform
// puts x[i] = C * x[i..i+7*skip]
// ##############################################################

int LA8(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=8;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels==0) return(1);
  switch(sense){
  case FWD:
    return(fLA8(x,len,1,nlevels));
  case BWD:
    return(bLA8(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

int fLA8(real* x, uint len, uint skip, uint nlevels){
  real tmp;
  uint i;
  // static uint opsf = 0;
  // static uint loops = 0;
  // uint i, i2, i3;
  
  if(skip < (1 << nlevels)){
    
    //store first 6 entries of vector
    real x0=x[0], xskip=x[skip], x2skip=x[2*skip], x3skip=x[3*skip],
      x4skip=x[4*skip], x5skip=x[5*skip];

    //do loop from i=0 -> i < len-3*skip : does transform to store in i=0 -> i=len-2*skip
    for(i=0;i<=(len - (skip<<3));i+=(skip<<1)){
      // printf("\nlen=%i,skip=%i,i=%i,i+skip=%i,i+2skip=%i,i+3skip=%i",len,skip,i,i+skip,i+2*skip,i+3*skip); 
      
      tmp = x[i]*C7 - x[i+skip]*C6 + x[i+2*skip]*C5 - x[i+3*skip]*C4 + x[i+4*skip]*C3 - x[i+5*skip]*C2 + x[i+6*skip]*C1 - x[i+7*skip]*C0; //detail
      x[i] = x[i]*C0 + x[i+skip]*C1 + x[i+2*skip]*C2 + x[i+3*skip]*C3 + x[i+4*skip]*C4 + x[i+5*skip]*C5 + x[i+6*skip]*C6 + x[i+7*skip]*C7; //scaling
      x[i+skip] = tmp;
      // opsf+=14;
      // loops+=1;
    }

    // do the last 3 iterations manually!

    i=len-6*skip; // should already be the case, but just making sure!
    tmp = x[i]*C7 - x[i+skip]*C6 + x[i+2*skip]*C5 - x[i+3*skip]*C4 + x[i+4*skip]*C3 - x[i+5*skip]*C2 + x0*C1 - xskip*C0; //detail
    x[i] = x[i]*C0 + x[i+skip]*C1 + x[i+2*skip]*C2 + x[i+3*skip]*C3 + x[i+4*skip]*C4 + x[i+5*skip]*C5 + x0*C6 + xskip*C7; //scaling
    x[i+skip] = tmp;
    
    i+=(skip<<1); //len-4*skip;
    tmp = x[i]*C7 - x[i+skip]*C6 + x[i+2*skip]*C5 - x[i+3*skip]*C4 + x0*C3 - xskip*C2 + x2skip*C1 - x3skip*C0; //detail
    x[i] = x[i]*C0 + x[i+skip]*C1 + x[i+2*skip]*C2 + x[i+3*skip]*C3 + x0*C4 + xskip*C5 + x2skip*C6 + x3skip*C7; //scaling
    x[i+skip] = tmp;

    i+=(skip<<1); //len-2*skip;
    tmp = x[i]*C7 - x[i+skip]*C6 + x0*C5 - xskip*C4 + x2skip*C3 - x3skip*C2 + x4skip*C1 - x5skip*C0; //detail
    x[i] = x[i]*C0 + x[i+skip]*C1 + x0*C2 + xskip*C3 + x2skip*C4 + x3skip*C5 + x4skip*C6 + x5skip*C7; //scaling
    x[i+skip] = tmp;

    // opsf+=;
    // loops+=1;
    // printvec(x,len);
    
    return(fLA8(x,len,skip<<1,nlevels));
  }  
  // printf("\nOpsf = %u",opsf);
  // printf("\nLoops = %u",loops);
  return(0);
}

int bLA8(real* x, uint len, uint skip){
  real tmp;
  // uint i, i2, i3;
  uint i;
  
  if(skip > 0){
        
    //store 6 entries of vector
    real xlskip=x[len-skip], xl2skip=x[len-2*skip], xl3skip=x[len-3*skip], xl4skip=x[len-4*skip], xl5skip=x[len-5*skip], xl6skip=x[len-6*skip];

    for(i=len-2*skip;i>=(6*skip);i-=(skip<<1)){

      tmp = x[i-6*skip]*C7 - x[i-5*skip]*C0 + x[i-4*skip]*C5 - x[i-3*skip]*C2 + x[i-2*skip]*C3 - x[i-skip]*C4 + x[i]*C1 - x[i+skip]*C6; //detail
      x[i] = x[i-6*skip]*C6 + x[i-5*skip]*C1 + x[i-4*skip]*C4 + x[i-3*skip]*C3 + x[i-2*skip]*C2 + x[i-skip]*C5 + x[i]*C0 + x[i+skip]*C7; //scaling
      x[i+skip] = tmp;
    }
    
    i=4*skip; // should already be the case, but just making sure!
    tmp = xl2skip*C7 - xlskip*C0 + x[i-4*skip]*C5 - x[i-3*skip]*C2 + x[i-2*skip]*C3 - x[i-skip]*C4 + x[i]*C1 - x[i+skip]*C6; //detail
    x[i] = xl2skip*C6 + xlskip*C1 + x[i-4*skip]*C4 + x[i-3*skip]*C3 + x[i-2*skip]*C2 + x[i-skip]*C5 + x[i]*C0 + x[i+skip]*C7; //scaling
    x[i+skip] = tmp;
    
    i-=(skip<<1); //2*skip;
    tmp = xl4skip*C7 - xl3skip*C0 + xl2skip*C5 - xlskip*C2 + x[i-2*skip]*C3 - x[i-skip]*C4 + x[i]*C1 - x[i+skip]*C6; //detail
    x[i] = xl4skip*C6 + xl3skip*C1 + xl2skip*C4 + xlskip*C3 + x[i-2*skip]*C2 + x[i-skip]*C5 + x[i]*C0 + x[i+skip]*C7; //scaling
    x[i+skip] = tmp;

    i-=(skip<<1); //0;
    tmp = xl6skip*C7 - xl5skip*C0 + xl4skip*C5 - xl3skip*C2 + xl2skip*C3 - xlskip*C4 + x[i]*C1 - x[i+skip]*C6; //detail
    x[i] = xl6skip*C6 + xl5skip*C1 + xl4skip*C4 + xl3skip*C3 + xl2skip*C2 + xlskip*C5 + x[i]*C0 + x[i+skip]*C7; //scaling
    x[i+skip] = tmp;

    // printvec(x,len);
    
    return(bLA8(x,len,skip>>1));
  }
  return(0);
}

// // ##############################################################
// // Next, another serial LA8 transform that has *better* time alignment
// // by filtering indices above and below
// // puts x[i] = C * x[i-2*skip..i+5*skip]
// // ##############################################################

// // no no no no no
// // doesn't work for in-place calculations like this
// // better to just shift afterwards if need be maybe

// int LA8_m2p6(real* x, uint len, short int sense, uint nlevels){
//   // sense '1' is forwards, '0' is backwards, anything else is sideways
//   uint filterlength=8;
//   nlevels = check_len_levels(len,nlevels,filterlength);
//   switch(sense){
//   case FWD:
//     return(fLA8_m2p6(x,len,1,nlevels));
//   case BWD:
//     return(bLA8_m2p6(x,len,1<<(nlevels-1)));
//   default:
//     printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
//     return(1);
//   }
// }


// int fLA8_m2p6(real* x, uint len, uint skip, uint nlevels){
//   // better time-aligned version - puts x[i] = C*x[i-2*skip..i+5*skip]
//   real tmp;
//   uint i;
  
//   if(skip < (1 << nlevels)){
    
//     //store first 6 entries of vector
//     real x0=x[0], xskip=x[skip], x2skip=x[2*skip], x3skip=x[3*skip],
//       xlskip=x[len-skip], xl2skip=x[len-2*skip];

//     // do the first iteration manually!
    
//     i=0;
//     tmp = xl2skip*C7 - xlskip*C6 + x0*C5 - xskip*C4 + x2skip*C3 - x3skip*C2 + x[4*skip]*C1 - x[5*skip]*C0; //detail
//     x[0] = xl2skip*C0 + xlskip*C1 + x0*C2 + xskip*C3 + x2skip*C4 + x3skip*C5 + x[4*skip]*C6 + x[5*skip]*C7; //scaling
//     x[i+skip] = tmp;

//     for(i=2*skip;i<=(len - 6*skip);i+=(skip<<1)){
//       // printf("\nlen=%i,skip=%i,i=%i,i+skip=%i,i+2skip=%i,i+3skip=%i",len,skip,i,i+skip,i+2*skip,i+3*skip); 
      
//       tmp = x[i-2*skip]*C7 - x[i-skip]*C6 + x[i]*C5 - x[i+skip]*C4 + x[i+2*skip]*C3 - x[i+3*skip]*C2 + x[i+4*skip]*C1 - x[i+5*skip]*C0; //detail
//       x[i] = x[i-2*skip]*C0 + x[i-skip]*C1 + x[i]*C2 + x[i+skip]*C3 + x[i+2*skip]*C4 + x[i+3*skip]*C5 + x[i+4*skip]*C6 + x[i+5*skip]*C7; //scaling
//       x[i+skip] = tmp;
//     }

//     // do the last 2 iterations manually!

//     i=len-4*skip; // should already be the case, but just making sure!
//     tmp = x[i-2*skip]*C7 - x[i-skip]*C6 + x[i]*C5 - x[i+skip]*C4 + x[i+2*skip]*C3 - x[i+3*skip]*C2 + x0*C1 - xskip*C0; //detail
//     x[i] = x[i-2*skip]*C0 + x[i-skip]*C1 + x[i]*C2 + x[i+skip]*C3 + x[i+2*skip]*C4 + x[i+3*skip]*C5 + x0*C6 + xskip*C7; //scaling
//     x[i+skip] = tmp;
    
//     i+=(skip<<1); //len-2*skip;
//     tmp = x[i-2*skip]*C7 - x[i-skip]*C6 + x[i]*C5 - x[i+skip]*C4 + x0*C3 - xskip*C2 + x2skip*C1 - x3skip*C0; //detail
//     x[i] = x[i-2*skip]*C0 + x[i-skip]*C1 + x[i]*C2 + x[i+skip]*C3 + x0*C4 + xskip*C5 + x2skip*C6 + x3skip*C7; //scaling
//     x[i+skip] = tmp;
    
//     return(fLA8_m2p6(x,len,skip<<1,nlevels));
//   }  
//   return(0);
// }


// int bLA8_m2p6(real* x, uint len, uint skip){
//   real tmp;
//   // uint i, i2, i3;
//   uint i;
  
//   if(skip > 0){
        
//     //store 6 entries of vector
//     real xlskip=x[len-skip], xl2skip=x[len-2*skip], xl3skip=x[len-3*skip], xl4skip=x[len-4*skip],
//       x0=x[0], xskip=x[skip];
    
//     // first iteration first.

//     i=len-2*skip;
//     tmp = x[i-4*skip]*C7 - x[i-3*skip]*C0 + x[i-2*skip]*C5 - x[i-skip]*C2 + x[i]*C3 - x[i+skip]*C4 + x0*C1 - xskip*C6; //detail
//       x[i] = x[i-4*skip]*C6 + x[i-3*skip]*C1 + x[i-2*skip]*C4 + x[i-skip]*C3 + x[i]*C2 + x[i+skip]*C5 + x0*C0 + xskip*C7; //scaling
//       x[i+skip] = tmp;

//     for(i=len-4*skip;i>=(4*skip);i-=(skip<<1)){

//       tmp = x[i-4*skip]*C7 - x[i-3*skip]*C0 + x[i-2*skip]*C5 - x[i-skip]*C2 + x[i]*C3 - x[i+skip]*C4 + x[i+2*skip]*C1 - x[i+3*skip]*C6; //detail
//       x[i] = x[i-4*skip]*C6 + x[i-3*skip]*C1 + x[i-2*skip]*C4 + x[i-skip]*C3 + x[i]*C2 + x[i+skip]*C5 + x[i+2*skip]*C0 + x[i+3*skip]*C7; //scaling
//       x[i+skip] = tmp;
//     }
    
//     i=2*skip; // should already be the case, but just making sure!
//     tmp = xl2skip*C7 - xlskip*C0 + x0*C5 - xskip*C2 + x[i]*C3 - x[i+skip]*C4 + x[i+2*skip]*C1 - x[i+3*skip]*C6; //detail
//     x[i] = xl2skip*C6 + xlskip*C1 + x0*C4 + xskip*C3 + x[i]*C2 + x[i+skip]*C5 + x[i+2*skip]*C0 + x[i+3*skip]*C7; //scaling
//     x[i+skip] = tmp;

//     i=0;
//     tmp = xl4skip*C7 - xl3skip*C0 + xl2skip*C5 - xlskip*C2 + x0*C3 - xskip*C4 + x[i+2*skip]*C1 - x[i+2*skip]*C6; //detail
//     x[i] = xl4skip*C6 + xl3skip*C1 + xl2skip*C4 + xlskip*C3 + x0*C2 + xskip*C5 + x[i+2*skip]*C0 + x[i+3*skip]*C7; //scaling
//     x[i+skip] = tmp;

//     // printvec(x,len);
    
//     return(bLA8_m2p6(x,len,skip>>1));
//   }
//   return(0);
// }

// ##############################################################
// Now, the lifting LA8 version, which should be faster!
// but by default has very wrong time alignment
// puts x[i] = C * x[i..i+7*skip]
// and x[i+1] = rev(C) * x[i-6*skip...i+skip]
// ##############################################################

int lLA8(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=8;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels==0) return(1);
  switch(sense){
  case FWD:
    return(flLA8(x,len,1,nlevels));
  case BWD:
    return(blLA8(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}

// int flLA8_wrong(real*x, uint len, uint skip, uint nlevels){
//   // this is what we believed would work from the maths
//   // but it ends up convoluting the wrong indices
//   uint i;
//   // with periodic boundary conditions

//   if(skip < (1 << nlevels)){
    
//     // first lifting loop
//     for(i=0;i<(len-2*skip);i+=(skip<<1)){
//       x[i+skip] = x[i+skip] + CL0*x[i] + CL1*x[i+2*skip];
//       // d1[l] = x[2l+1] + q11*x[2l] + q12*x[2l+2]
//     }
//     i=len-2*skip; // might delete!
//     x[i+skip] = x[i+skip] + CL0*x[i] + CL1*x[0];
    
//     // second lifting loop
//     for(i=0;i<(len-2*skip);i+=(skip<<1)){
//       x[i] = x[i] + CL2*x[i+skip] + CL3*x[i+3*skip];
//       // s1[l] = x[2l] + q21*d1[l] + q22*d1[l+1]
//     }
//     //i=len-2*skip; // might delete!
//     x[i] = x[i] + CL2*x[i+skip] + CL3*x[skip];

//     // third lifting loop
//     for(i=0;i<(len-2*skip);i+=(skip<<1)){
//       x[i+skip] = x[i+skip] + CL4*x[i] + CL5*x[i+2*skip];
//       // d2[l] = d1[l] + q31*s1[l] + q32*s1[l+1]
//     }
//     //i=len-2*skip; // might delete!
//     x[i+skip] = x[i+skip] + CL4*x[i] + CL5*x[0];

//     // fourth lifting loop
//     i=0;
//     x[i] = x[i] + CL6*x[len-skip] + CL7*x[i+skip];
//     for(i=(skip<<1);i<len;i+=(skip<<1)){
//       x[i] = x[i] + CL6*x[i-skip] + CL7*x[i+skip];
//       // s2[l] = s1[l] + q41*d2[l-1] + q42*d2[l]
//     }
    
//     // fifth lifting loop (aka the lifting step)
//     // first two iterations are done manually
//     i=0;
//     x[i+skip] = x[i+skip] + CL8*x[len-4*skip] + CL9*x[len-2*skip] + CL10*x[0];
    
//     i=(skip<<1);
//     x[i+skip] = x[i+skip] + CL8*x[len-2*skip] + CL9*x[0] + CL10*x[i];
    
//     for(i=(skip<<2);i<len;i+=(skip<<1)){
//       x[i+skip] = x[i+skip] + CL8*x[i-4*skip] + CL9*x[i-2*skip] + CL10*x[i];
//       // d3[l] = d2[l] + s1*K^2*s2[l-2] + s2*K^2*s2[l-1] + s3*K^2*s2[l]
//     }
    
//     // sixth lifting loop
//     for(i=0;i<len;i+=(skip<<1)){
//       x[i] = CL11*x[i];
//       //s3[l] = (K)*s2[l]
//       x[i+skip] = CL12*x[i+skip];
//       // d4[l] = (1/K)*d3[l]
//     }
    
//     return(flLA8(x,len,skip<<1,nlevels));
//   }

//   return(0);
// }


int flLA8(real*x, uint len, uint skip, uint nlevels){
  uint i;
  real switchsd;
  // with periodic boundary conditions

  if(skip < (1 << nlevels)){
    
    x[0] = x[0] + CL0*x[len-skip] + CL1*x[skip];
    // first lifting loop
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] + CL0*x[i-skip] + CL1*x[i+skip];
      // d1[l] = x[2l+1] + q11*x[2l] + q12*x[2l+2]
    }
    // printf("\n## cpp lifting -- 1st loop done ##");
    // printvec(x,len);
    
    x[len-skip] = x[len-skip] + CL2*x[0] + CL3*x[2*skip];
    // second lifting loop
    for(i=2*skip;i<(len-2*skip);i+=(skip<<1)){
      x[i-skip] = x[i-skip] + CL2*x[i] + CL3*x[i+2*skip];
      // s1[l] = x[2l] + q21*d1[l] + q22*d1[l+1]
    }
    x[len-3*skip] = x[len-3*skip] + CL2*x[len-2*skip] + CL3*x[0];
    // printf("\n## cpp lifting -- 2nd loop done ##");
    // printvec(x,len);

    x[0] = x[0] + CL4*x[len-skip] + CL5*x[skip];
    // third lifting loop
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] + CL4*x[i-skip] + CL5*x[i+skip];
      // d2[l] = d1[l] + q31*s1[l] + q32*s1[l+1]
    }
    // printf("\n## cpp lifting -- 3rd loop done ##");
    // printvec(x,len);

    // fourth lifting loop
    x[len-skip] = x[len-skip] + CL6*x[len-2*skip] + CL7*x[0];
    for(i=(skip<<1);i<len;i+=(skip<<1)){
      x[i-skip] = x[i-skip] + CL6*x[i-2*skip] + CL7*x[i];
      // s2[l] = s1[l] + q41*d2[l-1] + q42*d2[l]
    }
    // printf("\n## cpp lifting -- 4th loop done ##");
    // printvec(x,len);

    
    // fifth lifting loop (aka the lifting step)
    // first three iterations are done manually!
    x[0] = x[0] + CL8*x[len-5*skip] + CL9*x[len-3*skip] + CL10*x[len-skip];    
    x[2*skip] = x[2*skip] + CL8*x[len-3*skip] + CL9*x[len - skip] + CL10*x[skip];
    x[4*skip] = x[4*skip] + CL8*x[len-skip] + CL9*x[skip] + CL10*x[3*skip];
    
    for(i=skip*6;i<len;i+=(skip<<1)){
      x[i] = x[i] + CL8*x[i-5*skip] + CL9*x[i-3*skip] + CL10*x[i-skip];
      // d3[l] = d2[l] + s1*K^2*s2[l-2] + s2*K^2*s2[l-1] + s3*K^2*s2[l]
    }
    // printf("\n## cpp lifting -- 5th loop done ##");
    // printvec(x,len);

    
    // sixth lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      switchsd = CL12*x[i];
      //s3[l] = (K)*s2[l]
      x[i] = CL11*x[i+skip];
      // d4[l] = (1/K)*d3[l]
      x[i+skip] = switchsd;
    }
    // printf("\n## cpp lifting -- 6th loop done ##");
    // printvec(x,len);
  

    return(flLA8(x,len,skip<<1,nlevels));
  }

  return(0);
}

int blLA8(real*x, uint len, uint skip){
  uint i;
  real switchsd;
  
  if(skip > 0){
      
    // sixth lifting loop
    for(i=0;i<len;i+=(skip<<1)){
      switchsd = CL12*x[i];
      x[i] = CL11*x[i+skip];
      // d3[l] = (K)*d4[l]
      x[i+skip] = switchsd;
      //s2[l] = (1/K)*s3[l]
    } 

    
    // printf("\n## cpp lifting (BWDs) -- 6th loop done ##");
    // printf("\n##len = %u, skip = %u##",len,skip);
    // printvec(x,len);

    
    // fifth lifting loop (aka the lifting step)
    // first three iterations are done manually!
    x[0] = x[0] - CL8*x[len-5*skip] - CL9*x[len-3*skip] - CL10*x[len-skip];    
    x[2*skip] = x[2*skip] - CL8*x[len-3*skip] - CL9*x[len - skip] - CL10*x[skip];
    x[4*skip] = x[4*skip] - CL8*x[len-skip] - CL9*x[skip] - CL10*x[3*skip];
    
    for(i=skip*6;i<len;i+=(skip<<1)){
      x[i] = x[i] - CL8*x[i-5*skip] - CL9*x[i-3*skip] - CL10*x[i-skip];
      // d2[l] = d3[l] - s1*K^2*s2[l-2] - s2*K^2*s2[l-1] - s3*K^2*s2[l]
    }

    // printf("\n## cpp lifting (BWDs) -- 5th loop done ##");
    // printf("\n##len = %u, skip = %u##",len,skip);
    // printvec(x,len);

    
    // fourth lifting loop
    x[len-skip] = x[len-skip] - CL6*x[len-2*skip] - CL7*x[0];
    for(i=(skip<<1);i<len;i+=(skip<<1)){
      x[i-skip] = x[i-skip] - CL6*x[i-2*skip] - CL7*x[i];
      // s1[l] = s2[l] - q41*d2[l-1] - q42*d2[l]
    }
    
    // printf("\n## cpp lifting (BWDs) -- 4th loop done ##");
    // printf("\n##len = %u, skip = %u##",len,skip);
    // printvec(x,len);


    x[0] = x[0] - CL4*x[len-skip] - CL5*x[skip];
    // third lifting loop
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] - CL4*x[i-skip] - CL5*x[i+skip];
      // d1[l] = d2[l] - q31*s1[l] - q32*s1[l+1]
    }

    // printf("\n## cpp lifting (BWDs) -- 3rd loop done ##");
    // printf("\n##len = %u, skip = %u##",len,skip);
    // printvec(x,len);
    

    x[len-skip] = x[len-skip] - CL2*x[0] - CL3*x[2*skip];
    // second lifting loop
    for(i=2*skip;i<(len-2*skip);i+=(skip<<1)){
      x[i-skip] = x[i-skip] - CL2*x[i] - CL3*x[i+2*skip];
      // s1[l] = x[2l] - q21*d1[l] - q22*d1[l+1]
    }
    x[len-3*skip] = x[len-3*skip] - CL2*x[len-2*skip] - CL3*x[0];

    // printf("\n## cpp lifting (BWDs) -- 2nd loop done ##");
    // printf("\n##len = %u, skip = %u##",len,skip);
    // printvec(x,len);


    x[0] = x[0] - CL0*x[len-skip] - CL1*x[skip];
    // first lifting loop
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] - CL0*x[i-skip] - CL1*x[i+skip];
      // d0[l] = d1[l] - q11*x[2l] - q12*x[2l+2]
    }

    // printf("\n## cpp lifting (BWDs) -- 1st loop done ##");
    // printf("\n##len = %u, skip = %u##",len,skip);
    // printvec(x,len);


    return(blLA8(x,len,skip>>1));
    }

  return(0);
}


int lompLA8(real* x, uint len, short int sense, uint nlevels){
  // sense '1' is forwards, '0' is backwards, anything else is sideways
  uint filterlength=8;
  nlevels = check_len_levels(len,nlevels,filterlength);
  if(nlevels==0) return(1);
  switch(sense){
  case FWD:
    return(flompLA8(x,len,1,nlevels));
  case BWD:
    return(blompLA8(x,len,1<<(nlevels-1)));
  default:
    printf("\nSense must be 1 for forward or 0 for backwards. We don't do sideways.\n");
    return(1);
  }
}


int flompLA8(real*x, uint len, uint skip, uint nlevels){
  uint i;
  real switchsd;
  // with periodic boundary conditions

  if(skip < (1 << nlevels)){
    
    x[0] = x[0] + CL0*x[len-skip] + CL1*x[skip];
    // first lifting loop
#pragma omp parallel for private (i)
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] + CL0*x[i-skip] + CL1*x[i+skip];
      // d1[l] = x[2l+1] + q11*x[2l] + q12*x[2l+2]
    }
    
    x[len-skip] = x[len-skip] + CL2*x[0] + CL3*x[2*skip];
    // second lifting loop
#pragma omp parallel for private (i)
    for(i=2*skip;i<(len-2*skip);i+=(skip<<1)){
      x[i-skip] = x[i-skip] + CL2*x[i] + CL3*x[i+2*skip];
      // s1[l] = x[2l] + q21*d1[l] + q22*d1[l+1]
    }
    x[len-3*skip] = x[len-3*skip] + CL2*x[len-2*skip] + CL3*x[0];

    x[0] = x[0] + CL4*x[len-skip] + CL5*x[skip];
    // third lifting loop
#pragma omp parallel for private (i)
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] + CL4*x[i-skip] + CL5*x[i+skip];
      // d2[l] = d1[l] + q31*s1[l] + q32*s1[l+1]
    }

    // fourth lifting loop
    x[len-skip] = x[len-skip] + CL6*x[len-2*skip] + CL7*x[0];
#pragma omp parallel for private (i)
    for(i=(skip<<1);i<len;i+=(skip<<1)){
      x[i-skip] = x[i-skip] + CL6*x[i-2*skip] + CL7*x[i];
      // s2[l] = s1[l] + q41*d2[l-1] + q42*d2[l]
    }
    
    // fifth lifting loop (aka the lifting step)
    // first three iterations are done manually!
    x[0] = x[0] + CL8*x[len-5*skip] + CL9*x[len-3*skip] + CL10*x[len-skip];    
    x[2*skip] = x[2*skip] + CL8*x[len-3*skip] + CL9*x[len - skip] + CL10*x[skip];
    x[4*skip] = x[4*skip] + CL8*x[len-skip] + CL9*x[skip] + CL10*x[3*skip];
    
#pragma omp parallel for private (i)
    for(i=skip*6;i<len;i+=(skip<<1)){
      x[i] = x[i] + CL8*x[i-5*skip] + CL9*x[i-3*skip] + CL10*x[i-skip];
      // d3[l] = d2[l] + s1*K^2*s2[l-2] + s2*K^2*s2[l-1] + s3*K^2*s2[l]
    }
    
    // sixth lifting loop
#pragma omp parallel for private (i,switchsd)
    for(i=0;i<len;i+=(skip<<1)){
      switchsd = CL12*x[i];
      //s3[l] = (K)*s2[l]
      x[i] = CL11*x[i+skip];
      // d4[l] = (1/K)*d3[l]
      x[i+skip] = switchsd;
    }
    
    return(flompLA8(x,len,skip<<1,nlevels));
  }

  return(0);
}

int blompLA8(real*x, uint len, uint skip){
  uint i;
  real switchsd;
  
  if(skip > 0){
      
    // sixth lifting loop
#pragma omp parallel for private (i,switchsd)
    for(i=0;i<len;i+=(skip<<1)){
      switchsd = CL12*x[i];
      x[i] = CL11*x[i+skip];
      // d3[l] = (K)*d4[l]
      x[i+skip] = switchsd;
      //s2[l] = (1/K)*s3[l]
    } 
    
    // fifth lifting loop (aka the lifting step)
    // first three iterations are done manually!
    x[0] = x[0] - CL8*x[len-5*skip] - CL9*x[len-3*skip] - CL10*x[len-skip];    
    x[2*skip] = x[2*skip] - CL8*x[len-3*skip] - CL9*x[len - skip] - CL10*x[skip];
    x[4*skip] = x[4*skip] - CL8*x[len-skip] - CL9*x[skip] - CL10*x[3*skip];
#pragma omp parallel for private (i)
    for(i=skip*6;i<len;i+=(skip<<1)){
      x[i] = x[i] - CL8*x[i-5*skip] - CL9*x[i-3*skip] - CL10*x[i-skip];
      // d2[l] = d3[l] - s1*K^2*s2[l-2] - s2*K^2*s2[l-1] - s3*K^2*s2[l]
    }
    
    // fourth lifting loop
    x[len-skip] = x[len-skip] - CL6*x[len-2*skip] - CL7*x[0];
#pragma omp parallel for private (i)
    for(i=(skip<<1);i<len;i+=(skip<<1)){
      x[i-skip] = x[i-skip] - CL6*x[i-2*skip] - CL7*x[i];
      // s1[l] = s2[l] - q41*d2[l-1] - q42*d2[l]
    }
    
    x[0] = x[0] - CL4*x[len-skip] - CL5*x[skip];
    // third lifting loop
#pragma omp parallel for private (i)
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] - CL4*x[i-skip] - CL5*x[i+skip];
      // d1[l] = d2[l] - q31*s1[l] - q32*s1[l+1]
    }

    x[len-skip] = x[len-skip] - CL2*x[0] - CL3*x[2*skip];
    // second lifting loop
#pragma omp parallel for private (i)
    for(i=2*skip;i<(len-2*skip);i+=(skip<<1)){
      x[i-skip] = x[i-skip] - CL2*x[i] - CL3*x[i+2*skip];
      // s1[l] = x[2l] - q21*d1[l] - q22*d1[l+1]
    }
    x[len-3*skip] = x[len-3*skip] - CL2*x[len-2*skip] - CL3*x[0];

    x[0] = x[0] - CL0*x[len-skip] - CL1*x[skip];
    // first lifting loop
#pragma omp parallel for private (i)
    for(i=2*skip;i<len;i+=(skip<<1)){
      x[i] = x[i] - CL0*x[i-skip] - CL1*x[i+skip];
      // d0[l] = d1[l] - q11*x[2l] - q12*x[2l+2]
    }

    return(blompLA8(x,len,skip>>1));
    }

  return(0);
}
