#ifndef THRESH_H
#define THRESH_H

#include "utils.h"
#include "wvtheads.h"
#include "transform.h"

//


void threshold_dwt(real* in, real* out, uint len, real thresh, short hardness);
// thresholding function for vectors thresholded over all levels
// assumes a full transform

void threshold_dwt(real* in, real* out, uint len, real thresh, short hardness, uint minlevel, uint maxlevel, uint levels);
// thresholding function for vectors thresholded over levels in specified range

void threshold_modwt(real* in, real* out, uint len, real thresh, short hardness, short modwttype, uint levels);
// thresholding function for vectors thresholded over all levels

void threshold_modwt(real* in, real* out, uint len, real thresh, short hardness, short modwttype, uint minlevel, uint maxlevel, uint levels);
// thresholding function for vectors thresholded over levels in specified range


void threshold(wst* win, wst* wout, real thresh, short hardness);
void threshold(wst* win, wst* wout, real thresh, short hardness, uint minlevel, uint maxlevel);
// wrappers for wst types to execute above functions

real thresh_coef(real coef, real thresh, short hardness);
// function used by the above for thresholding a coefficient according to a scheme as defined by 'hardness'

int null_level(real* x, uint len, short level);
real univ_thresh(wst*w, uint minlevel, uint maxlevel);
// allows calculation of universal threshold over multiple levels
// standard usage (actual univ thresh defn) univ_thresh(w,0,0)

real CVT(wst *w, short hardness, real tol, uint minlevel, uint maxlevel);
// cross validation thresholding - finds & uses the threshold that minimises the MSE when comparing interpolated thresholded even/odds to original (noisy) odds/evens.

real interp_mse(wst* wn, wst* ye, wst* yo);
// calculates interpolation error
// comparing noisy wn to
// smoothed ye & yo
// i.e. interpolate ye & compare to odd values in w
// & yo with even w

#endif //ifndef
