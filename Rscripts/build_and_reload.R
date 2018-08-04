## Rcpp::compileAttributes()
devtools::build()

tmpLibLoc <- paste0(getwd(),"/../templib")
tmpLibArg <- paste0("--library=",tmpLibLoc)
devtools::install(args = tmpLibArg)

library("wavecuda",lib.loc="../templib")

