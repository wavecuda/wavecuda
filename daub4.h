#ifndef DAUB4_H
#define DAUB4_H

#include "wvtheads.h"

int Daub4(real* x, uint len, short int sense, uint nlevels);
int fDaub4(real* x, uint len, uint skip, uint nlevels);
int bDaub4(real* x, uint len, uint skip);

/* int threshold(real* x, uint len, real thresh, short hardness); */
/* int null_level(real* x, uint len, short level); */
/* real univ_thresh(real*x, uint len); */


int lDaub4(real* x, uint len, short int sense, uint nlevels); //lifted
int flDaub4(real* x, uint len, uint skip, uint nlevels);
int blDaub4(real* x, uint len, uint skip);

int l2Daub4(real* x, uint len, short int sense, uint nlevels); //lifted, 2
int fl2Daub4(real* x, uint len, uint skip, uint nlevels);
int bl2Daub4(real* x, uint len, uint skip);


int lompDaub4(real* x, uint len, short int sense, uint nlevels); //lifted, omp
int flompDaub4(real* x, uint len, uint skip, uint nlevels);
int blompDaub4(real* x, uint len, uint skip);

int Daub4MODWTpo(real* x, real* xdat, uint len, short int sense, uint nlevels);
int fDaub4MODWTpo(real* x, real* xdat, uint len, uint skip, uint nlevels);
int bDaub4MODWTpo(real* x, real* xdat, uint len, uint skip, uint nlevels);

int Daub4MODWTto(real* x, real* xdat, uint len, short int sense, uint nlevels);
int fDaub4MODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels);
int bDaub4MODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels);

int fDaub4SDout(real* Xin, real* Sout, real* Dout, uint len, uint incri, uint skipi, uint skipo, uint nlevels);
int bDaub4SDout(real* Xout, real* Sin, real* Din, uint len, uint incri, uint skipi, uint skipo, uint nlevels);

int lDaub4MODWTpo(real* x, real* xdat, uint len, short int sense, uint nlevels);
int flDaub4MODWTpo(real* x, real* xdat, uint len, uint skip, uint nlevels);
int blDaub4MODWTpo(real* x, real* xdat, uint len, uint skip, uint nlevels);

int lDaub4MODWTto(real* x, real* xdat, uint len, short int sense, uint nlevels);
int flDaub4MODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels);
int blDaub4MODWTto(real* x, real* xdat, uint len, uint skip, uint nlevels);

int flDaub4SDout(real* Xin, real* Sout, real* Dout, uint len, uint incri, uint skipi, uint skipo, uint nlevels);
int blDaub4SDout(real* Xout, real* Sin, real* Din, uint len, uint incri, uint skipi, uint skipo, uint nlevels);


#endif //ifndef
