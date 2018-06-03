source("rfunctions.R")

library(wavethresh)
library(wmtsa)
library(waveslim)


cat("\nComparison of times for FWD MODWT using Haar filter of vector of length 2^i using different packages")
cat("\n----------------------------------------------------------------\n")

cat("\n(1) : wavecuda CPU transform (serial)")
cat("\n(2) : wavecuda GPU transform")
cat("\n(3) : waveslim transform")
cat("\n(4) : wavethresh transform")
cat("\n(5) : wmtsa transform")

cat("\nNB GPU time, (3), includes transform time of vector from CPU memory to GPU memory\n")

######################################################################

cat("\n\t(1)\t(2)\t(3)\t(4)\t(5)\n")


for(i in (10:20)){

    cat("----------------------------------------------------")
    cat("\ni=",i,"\t")
    x <- as.numeric(rnorm(n=2^i,mean=0,sd=10))

    cat(system.time(
        CPUTransform(x,"FWD",i,"MODWT.TO","Haar")
    )["elapsed"],"\t")

    cat(system.time(
        GPUTransform(x,"FWD",i,"MODWT.TO","Haar")
    )["elapsed"],"\t")

    cat(system.time(
        modwt(x,wf="haar",n.levels=i)
    )["elapsed"],"\t")

    cat(system.time(
        wd(x,filter.number=1,family="DaubExPhase",type="station")
    )["elapsed"],"\t")

    cat(system.time(
        wavMODWT(x,wavelet="haar",n.levels=i)
    )["elapsed"],"\n")

}
cat("----------------------------------------------------\n")


######################################################################
