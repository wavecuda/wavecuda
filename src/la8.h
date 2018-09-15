#ifndef LA8_H
#define LA8_H

#include "wvtheads.h"

int LA8(real* x, uint len, short int sense, uint nlevels);
int fLA8(real* x, uint len, uint skip, uint nlevels);
int bLA8(real* x, uint len, uint skip);

int lLA8(real* x, uint len, short int sense, uint nlevels);
int flLA8(real* x, uint len, uint skip, uint nlevels);
int blLA8(real* x, uint len, uint skip);

int lompLA8(real* x, uint len, short int sense, uint nlevels);
int flompLA8(real* x, uint len, uint skip, uint nlevels);
int blompLA8(real* x, uint len, uint skip);



#endif //ifndef
