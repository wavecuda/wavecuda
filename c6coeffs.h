#ifndef C6_COEFFS_H
#define C6_COEFFS_H

// The standard wavelet coefficients:

#define C0 -7.273261951252645e-2 // sqrt(2)*(1 - sqrt(7))/32
#define C1  3.378976624574818e-1 // sqrt(2)*(5 + sqrt(7))/32
#define C2  8.525720202116004e-1 // sqrt(2)*(7 + sqrt(7))/16
#define C3  3.848648468648577e-1 // sqrt(2)*(7 - sqrt(7))/16
#define C4 -7.273261951252645e-2 // sqrt(2)*(1 - sqrt(7))/32
#define C5 -1.565572813579199e-2 // sqrt(2)*(-3+ sqrt(7))/32

// The lifting wavelet coefficients

#define CL0   4.645751311064591e+0 // q11
#define CL1   1.673667737846021e-2 // q21
#define CL2  -4.861001748086121e+0 // q31
#define CL3   3.160743211769211e-1 // q41
#define CL4   3.857229635307634e-1 // q42
#define CL5  -2.460004994531774e-1 // s*K^2
#define CL6   1.069044967649698e+0 // K
#define CL7   9.354143466934853e-1 // 1/K
// these coefficients don't need brackets in the macro definitions - I checked!

#endif //ifndef
