#ifndef C6_H
#define C6_H

#include "wvtheads.h"
//#include "wavutils.h"

int C6(real* x, uint len, short int sense, uint nlevels);
int fC6(real* x, uint len, uint skip, uint nlevels);
int bC6(real* x, uint len, uint skip);

int lC6(real* x, uint len, short int sense, uint nlevels);
int flC6(real* x, uint len, uint skip, uint nlevels);
int blC6(real* x, uint len, uint skip);

int lompC6(real* x, uint len, short int sense, uint nlevels);
int flompC6(real* x, uint len, uint skip, uint nlevels);
int blompC6(real* x, uint len, uint skip);



#endif //ifndef
