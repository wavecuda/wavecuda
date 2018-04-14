#ifndef WAVUTILS_H
#define WAVUTILS_H

#include "wvtheads.h"
#include "utils.h"

#define CTI 0 // Coalesced to Interleaved
#define ITC 1 // Interleaved to Coalesced

#define MINCOL 10

int rejig(real* x, uint len, short int sense);
// Rejig a vector x of length len between coalesced & interleaved layout
// Rejig in the sense of 'sense'

uint check_len_levels(uint len, uint nlevels, uint filterlength);
// checks that we are entering a valid len & number of levels.
// returns 0 if either are invalid
// else returns the validated nlevels

uint check_len_levels(uint len, uint nlevels, uint minlevel, uint maxlevel, uint filterlength);
// augmented levels checker
// runs above check
// then verifies that the min & max level are appropriately set

void printwavlevel(real* x, uint len, uint level);
//prints detail coefficients at specified level
//useful debugging tool!

void shift_wvt_vector(real* x, uint len, uint skip, int dshift, int sshift);
// shift wvt coeffs around
// use dshift (sshift) to shift detail (scaling) coeffs
// to the right by the shift amount
// to improve the time alignment

void print_modwt_vec_po(real* x, uint len, uint nlevels);
// prints a packet ordered modwt vector
// in a nice, user friendly way

void print_modwt_vec_to(real* x, uint len, uint nlevels);
// prints a time ordered modwt vector
// in a nice, user friendly way

void ab_bin(char *ab, uint shift, uint l2s, uint nlevels);
// function used by po modwt printing function
// fills helpful strings with "a" & "b" characters
// to label the packets

void shift_loop(real* x, uint len, uint skip, uint ushift, int sign, uint i, real* array, uint* ai);
// function to be called in the loop

void ax_wvt_vector(real *x, uint len, uint skip, real da, real sa);
// multiply wvt coeffs by scalars
// for a particular value of skip
// da for the detail
// sa for the scaling

void writeavpackets(real* xr, real* xp0, real* xp1, uint len, uint skipxr);
// writes xr with the average of the scaling coefficients from xp0 & xp1,
// which are assumed to have been reconstructed one level
// if no thresholding has been done, then each set of scaling coefficients
// in all 3 vectors should be identical

void write_test_modwt(real* xm, uint len, uint nlevels);
// writes to xm, a modwt data vector
// writes helpful values for testing!

int cmpmodwtlevelto(real* v1, real* v2, uint len, uint level, int numerrors);
// wrapper to cmpvec that tests levels of a modwt time ordered data vector

int cmpmodwtlevelto(real* v1, real* v2, uint len, int numerrors);
// called by cmpmodwt, does work for TO MODWT, wrapper to cmpvec

int cmpmodwtlevelpo(real* v1, real* v2, uint len, uint l, int numerrors);
// called by cmpmodwt, does work for PO MODWT, wrapper to cmpvec

int cmpmodwt(wst* w1, wst* w2, int level, int numerrors);
// wrapper to cmpvec that tests levels of modwt wavelets
// taking wavelet types
// will even compare Packet Ordered vs Time Ordered (main reason for writing!)
// and gives helpful info as to location of errors
// negative "level" argument will test all levels

int convert_modwt(wst* w);
// function converts a modwt wst structure
// from PO -> TO
// or from TO -> PO
// depending on the type of the input

int convert_modwt_level(real* xin, real* xout, uint l, short intype, uint len);
// converts a vector of a level of 'len' modwt coefficients
// from intype to not intype
// called by convert_modwt and also cmpmodwt

wst* create_wvtstruct(short ttype, short filt, short filtlen, uint levels, uint len);
// malloc wvtstruct components, return pointer to wst type

wst* create_wvtstruct(real* x, short ttype, short filt, short filtlen, uint levels, uint len);
// creates wvstruct using given input vector, malloc xmod component if a MODWT type, return pointer to wst type

wst* create_wvtstruct(real* x, real* xmod, short ttype, short filt, short filtlen, uint levels, uint len);
// creates modwt wvstruct using given x & xmod inputs, return pointer to wst type

wst* dup_wvtstruct(wst *wfrom);
// duplicate a wvtstruct
// returns the new wvtstruct

wst* dup_wvtstruct(wst *wfrom, short memcpy);
// deplicates a wvtstruct but with the option of not copying across
// all the elements in pointer arrays, using memcpy as a boolean

real* create_wvtbackup(wst *w, uint minlevel, uint maxlevel);
// just creates a backup of the necessary levels of transform
// i.e. those that will be overwritten in cross validation

void kill_wvtstruct(wst *w);
// free things that need freeing!

void remove_wvtstruct(wst *w);
// free the structure but leave the allocated bits

void isolate_dlevels(wst* w, uint minlevel, uint maxlevel, real* dvec, uint n);
// produce a vector of only detail coefficients from levels
// between minlevel & maxlevel
// and writes it to dvec which is already calculated to be size n

void print_wst_info(wst *w);
// print information about elements of wst type


uint ndetail_thresh(short ttype, uint len, uint minlevel, uint maxlevel);
// returns the number of detail coefficients inside thresholding levels

uint ndetail_thresh(wst* w, uint minlevel, uint maxlevel);
// wrapper to function above taking wst argument

short get_filt_len(short filt);
// returns filter length for a given filter

#endif //ifndef
