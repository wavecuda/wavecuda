#ifndef LA8_COEFFS_H
#define LA8_COEFFS_H

// The standard wavelet coefficients:

#define C0 -7.576571478950221e-2
#define C1 -2.963552764600249e-2
#define C2  4.976186676327750e-1
#define C3  8.037387518051321e-1
#define C4  2.978577956053061e-1
#define C5 -9.921954357663353e-2
#define C6 -1.260396726203130e-2
#define C7  3.222310060405147e-2

// The lifting wavelet coefficients

#define CL0  -5.254527458732594e+1 // q11
#define CL1  -2.556583965520256e+0 // q12
#define CL2   1.784231379329362e-2 // q21
#define CL3  -7.110054749942946e-4 // q22
#define CL4   8.411184074671726e+2 // q31
#define CL5   5.498294188059471e+2 // q32
#define CL6  -3.361060977406985e-3 // q41
#define CL7   2.289462248002221e-2 // q42
#define CL8  -8.434130678687068e-1 // s1*K^2
#define CL9  -2.942429943568852e+0 // s2*K^2
#define CL10 -4.367837909852400e+1 // s3*K^2
#define CL11  1.408227631773041e+0 // K
#define CL12  7.101124686361548e-1 // 1/K
// these coefficients don't need brackets in the macro definitions - I checked!

#endif //ifndef
