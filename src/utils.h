#ifndef UTILS_H
#define UTILS_H

#include <time.h>
#include "wvtheads.h"

void initrandvec(real* x, uint len);
void copyvec(real* from, real* to, uint len);
void copyvecskip(real* from, uint skipf, uint lenf, real* to, uint skipt);
void copyvecskipshiftred(real* from, uint skipf, uint lenf, real* to, uint shift);
uint log2int(uint k);
int cmpvec(real* vector1, real* vector2, uint len);
int cmpvec(real* v1, real* v2, unsigned int len, real precision, real probzero, int numerrors);
void printvec(real* x, uint len);
void printvecskip(real* x, uint len,uint skip);
void printmat(real** x, uint nrow, uint ncol);
void printmatvec(real* x, uint nrow, uint ncol);
void axpby(real* x, real a, real* y, real b, real* res, uint len);

float timer(int T);
float mptimer(int T);

int compareReal (const void* x1, const void* x2);
real median(real* x, uint len);
// calculate median with an temporary array malloc
// to leave x alone

real median(real* x, uint len, short overwrite);
// calculate median with the option of overwriting x

real mad(real *x, uint len);

/* inline int min(int a, int b){ */
/*     if (a > b) */
/*       return(b); */
/*     return(a); */
/* } */

/* inline int max(int a, int b){ */
/*     if (a < b) */
/*       return(b); */
/*     return(a); */
/* } */
// already defined in CUDA math_functions

void read_1darray(char *s,real* x, uint len, uint skip);
// read in a 1d array
// from a file formatted with one value per line, no commas etc
// and writes to a real array that has already been allocated!
// with the option to skip
void read_1darray(char *s,real* x, uint len);
// no skip!

void write_1darray(char *s,real* x, uint len);
// writes a real array to a file given by s

uint reverse_bits(uint v, int length);

real sumvec(real* x, uint len);
// little function for debugging
// sums a vector!

#endif //ifndef
