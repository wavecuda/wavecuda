#ifndef D4_COEFFS_H
#define D4_COEFFS_H

// The standard wavelet coefficients:

#define C0  0.4829629131445341 // (1+sqrt(3))/(4*sqrt(2))
#define C1  0.8365163037378079 // (3+sqrt(3))/(4*sqrt(2))
#define C2  0.2241438680420134 // (3-sqrt(3))/(4*sqrt(2))
#define C3 -0.1294095225512604 // (1-sqrt(3))/(4*sqrt(2))

// The lifting (v1) wavelet coefficients

#define Cl0 1.732050807568877 //sqrt(3)
#define Cl1 0.433012701892219 //sqrt(3)/4
#define Cl2 -0.066987298107781 //(sqrt(3)-2)/4
#define Cl3 0.517638090205042 //(sqrt(3)-1)/sqrt(2)
#define Cl4 1.931851652578137 //(sqrt(3)+1)/sqrt(2)

// The lifting (v2) wavelet coefficients

#define Cl20 0.577350269189626 //1/sqrt(3)
#define Cl21 0.200961894323342 //(6-sqrt(3)*3)/4
#define Cl22 0.433012701892219 //sqrt(3)/4
#define Cl23 0.333333333333333 //1/3
#define Cl24 1.115355071650411 //(3+sqrt(3))/(3*sqrt(2))
#define Cl25 0.896575472168054 //(3-sqrt(3))/(3*sqrt(2))

#endif //ifndef
