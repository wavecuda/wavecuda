// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// RcpuTransform
NumericVector RcpuTransform(NumericVector x, NumericVector xmod, int len, int sense, int nlevels, int ttype, int filter, int filterlen);
RcppExport SEXP _wavecuda_RcpuTransform(SEXP xSEXP, SEXP xmodSEXP, SEXP lenSEXP, SEXP senseSEXP, SEXP nlevelsSEXP, SEXP ttypeSEXP, SEXP filterSEXP, SEXP filterlenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type xmod(xmodSEXP);
    Rcpp::traits::input_parameter< int >::type len(lenSEXP);
    Rcpp::traits::input_parameter< int >::type sense(senseSEXP);
    Rcpp::traits::input_parameter< int >::type nlevels(nlevelsSEXP);
    Rcpp::traits::input_parameter< int >::type ttype(ttypeSEXP);
    Rcpp::traits::input_parameter< int >::type filter(filterSEXP);
    Rcpp::traits::input_parameter< int >::type filterlen(filterlenSEXP);
    rcpp_result_gen = Rcpp::wrap(RcpuTransform(x, xmod, len, sense, nlevels, ttype, filter, filterlen));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_wavecuda_RcpuTransform", (DL_FUNC) &_wavecuda_RcpuTransform, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_wavecuda(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
