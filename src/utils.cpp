#include "utils.h"

// Copy a vector of length 'len' into another vector
// Both vectors must already be allocated
void copyvec(real* from, real* to, uint len){
  uint i;
  for(i=0;i<len;i++) to[i]=from[i];
}

// Copy a vector of length 'lenf' into another vector
// Both vectors must already be allocated
// copy indices as determined by skipf & skipt
void copyvecskip(real* from, uint skipf, uint lenf, real* to, uint skipt){
  uint i,j=0;
  for(i=0;i<lenf;i+=skipf){
    to[j]=from[i];
    j+=skipt;
  }
}

// same as function above
// but also shifts vector by shift
// and reduces in size
void copyvecskipshiftred(real* from, uint skipf, uint lenf, real* to, uint shift){
  uint i,j=0;
  for(i=0;i<lenf;i+=skipf){
    to[j]=from[(i + shift*skipf) % lenf];
    j++;
  }
}

// takes a positive integer
// & finds the integer log base 2
uint log2int(uint k){
  uint log=0;
  while(k>1){
    log++;k=k>>1;
  }
  return(log);
}

// Puts random integers into the first 'len' elements of an allocated vector
void initrandvec(real* x, unsigned int len){
  unsigned int i;
  for(i=0;i<len;i++) x[i]=rand();
}

// cmpvec...
// Compares the first 'len' elements of two vectors
// prints indexes where they differ
// returns #elts where they differ

// generic form without specifying precision/zero
int cmpvec(real* v1, real* v2, unsigned int len){
  real precision=1e-5, probzero=1e-10;
  int numerrors=10;
  return(cmpvec(v1,v2,len,precision,probzero,numerrors));
}

// NB, if probzero < |x| < precision
// and |y| < probzero
// then this will throw an error
// even if |x| is very small
// we can edit probzero if this is problematic

// more adaptible form.
int cmpvec(real* v1, real* v2, unsigned int len, real precision, real probzero, int numerrors){
  unsigned int i, errs=0;
  int sign,sign2;
  for(i=0;i<len;i++){
    sign=(v1[i]>=0) - (v1[i]<0);
    sign2=(v2[i]>=0) - (v2[i]<0);
    //if((!(v1[i]*sign>probzero))||(!(v2[i]*sign2>probzero))){ //else it's probably zero
    if(!((v1[i]*sign<probzero)&&(v2[i]*sign2<probzero))){ //else it's probably zero
      if((v1[i]*sign<v2[i]*sign2*(1-precision)) || (v1[i]*sign>v2[i]*sign2*(1+precision))){
	if(((numerrors>=0)&&(errs<(uint)numerrors))||(numerrors<0)){
	  // negative numerrors means print all errors
	  // positive numerrors means print up to numerrors
	  printf("Vectors differ at index %u. v1[%u] = %4.5e,\t v2[%u] = %4.5e\n",i,i,v1[i],i,v2[i]);
	}//if((num...
	errs+=1;
      }//if((v1...
    }//if(!(v1..
  }//for(...
  if(errs>0) printf("++Total number of measured errors = %u++\n",errs);
  // if(errs==0) printf("Congrats: no errors!\n");
  return(errs);
}


void printvec(real* x, uint len){
  uint i;
  printf("\nPrint vector...\n");
  for(i=0;i<len;i++) printf("[%u]\t%g\n",i,x[i]);
  printf("\n");
}

void printvecskip(real* x, uint len,uint skip){
  uint i;
  printf("\nPrint vector with skip=%u...\n",skip);
  for(i=0;i<len;i+=skip) printf("%g\n",x[i]);
  printf("\n");
}

void printmat(real** x, uint nrow, uint ncol){
  uint i,j;
  printf("\nPrint matrix...\n");
  for(i=0;i<nrow;i++){
    for(j=0;j<ncol;j++){
      printf("%g,",x[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void printmatvec(real* x, uint nrow, uint ncol){
  //prints a 1D matrix
  uint i,j;
  printf("\nPrint matrix...\n");
  for(i=0;i<nrow*ncol;i+=ncol){
    for(j=0;j<ncol;j++){
      printf("%g,",x[i+j]);
    }
    printf("\n");
  }
  printf("\n");
}

void axpby(real* x, real a, real* y, real b, real* res, uint len){
  uint i;
  for(i=0;i<len;i++) res[i]=a*x[i] + b*y[i];
}

float timer(int T){
  static double then=0.0;
  double now, diff;
  
  now=(double)clock()/((double)CLOCKS_PER_SEC);
  diff=now-then;
  if (T<0) then=now;

  return((float)diff);
}

float mptimer(int T){
  static double then=0.0;
  double now, diff;
  now=omp_get_wtime();
  // now = 0;
  // printf("\n####line commented to compile in windows!###\n");
  diff=now-then;
  if (T<0) then=now;

  return((float)diff);
}


int compareReal (const void* x1, const void* x2){
  // compare function for qsort
  if ( *(real*)x1 <  *(real*)x2 ) return -1;
  if ( *(real*)x1 == *(real*)x2 ) return 0;
  if ( *(real*)x1 >  *(real*)x2 ) return 1;
}

real median(real* x, uint len){
  return(median(x,len,0));
}

real median(real* x, uint len, short overwrite){
  real med;
  real* tmp;
  if(len==0){
    printf("\nError: cannot find median of vector length 0\n");
    return(0);
  }
  if(!overwrite){
    tmp = (real *)malloc(len*sizeof(real));
    copyvec(x,tmp,len);
  }
  else tmp = x;  
  qsort(tmp,len,sizeof(real),compareReal);
  med = 0.5*(tmp[(len-1)/2] + tmp[len/2]);
  if(!overwrite){
    free(tmp);
  }
  return(med);

}

real mad(real *x, uint len){
  real avmed;
  real* dev;
  real madev;
  uint i;
  dev = (real *)malloc(len*sizeof(real));
  avmed = median(x,len);
  for(i=0;i<len;i++) dev[i] = fabs(x[i] - avmed);
  // tmp now holds the absolute deviations
  madev = median(dev,len,1); // calculate median with overwriting allowed
  free(dev);
  return(1.4826*madev);
}

void read_1darray(char *s,real* x, uint len, uint skip){
  FILE *fp;
  uint i = 0, j = 0;
  double tmp; // temporary variable to read in as double
  // then we parse as (real) as necessary afterwards
  int ret;
  // return value of scanf. Should be 1 for each succesfully read value.
  fp = fopen(s, "r");
  // file is arranged just with a value on each line
  // from lines 1...len
  while(j<len){
    ret = fscanf(fp,"%lf",&tmp);
    if(ret != 1){
      printf("\nUnexpected value returnd by fscanf, %d, at i=%u\n",ret,i);
      break;
    }
    if(i % skip == 0){
      x[j] = (real)tmp;
      j++;
    }
    i++;
  }
  fclose(fp);
}

void read_1darray(char *s,real* x, uint len){
  // wrapper to version with skip
  read_1darray(s,x,len,1);
}

void write_1darray(char *s,real* x, uint len){
  FILE *fp;
  uint i = 0;
  double tmp; // temporary variable to write in as double
  int ret;
  fp = fopen(s, "w+");
  // writes to new file, overwrites if file already exists
  while(i<len){
    tmp = (double) x[i];
    ret = fprintf(fp,"%e\n",tmp);
    if(ret <= 0){
      printf("\nUnexpected value returnd by fprintf, %d, at i=%u\n",ret,i);
      break;
    }
    i++;
  }
  fclose(fp);
}

uint reverse_bits(uint v, const int length){
  // from  Bit Twiddling Hacks
  // By Sean Eron Anderson
  // seander@cs.stanford.edu
  // http://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
  
  // but edited!
  // length is number of bits we want to reverse
  // i.e. length = 1 => 0->0, 1->1
  //      length = 2 => 0->0, 1->2, 2->1, 3->3

  uint r = v & 1; // r will be reversed bits of v; first get LSB of v
  int s = length-1;
  
  
  while(v>>1)
    {   
      v>>=1;
      r <<= 1;
      r |= v & 1;
      s--;
    }
  r |= v & 1;
  if(s>0) r <<= s; // shift when v's highest bits are zero
  return(r);
}

real sumvec(real* x, uint len){
  // little function for debugging
  uint i;
  real sum=0;
  for(i = 0; i<len; i++) sum+=x[i];
  return(sum);
}
