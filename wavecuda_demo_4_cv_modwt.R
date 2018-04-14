source("rfunctions.R")

library(wavethresh)
library(wmtsa)
library(waveslim)


cat("\nComparison of times for MODWT CVT using Haar filter of vector of length 2^i using different packages")
cat("\n----------------------------------------------------------------\n")


cat("\n(1) : wavecuda CPU transform (serial)")
cat("\n(2) : wavecuda GPU transform")
cat("\n(3) : waveslim transform")
cat("\n(4) : wavethresh transform")
cat("\n(5) : wmtsa transform")

cat("\nNB GPU time, (3), includes transform time of vector from CPU memory to GPU memory\n")


cat("\n\t(1)\t(2)\t\t(4)\t\n")

for(i in (10:19)){

    cat("----------------------------------------------------")
    cat("\ni=",i,"\t")
    x <- as.numeric(rnorm(n=2^i,mean=0,sd=10))
    
    cat(system.time(
        CPUSmooth(xin=x,nlevels=i,transform.type="MODWT.TO",filter="Haar",thresh.type="cv",hard.soft="soft",min.level=1,max.level=i)
    )["elapsed"],"\t")

    cat(system.time(
        GPUSmooth(xin=x,nlevels=i,transform.type="MODWT.TO",filter="Haar",thresh.type="cv",hard.soft="soft",min.level=1,max.level=i)
    )["elapsed"],"\t")

    cat("\t")

    cat(system.time(
        wstCV1(x,ll=0,filter.number=1,family="DaubExPhase",type="soft")
    )["elapsed"],"\n")

}
cat("----------------------------------------------------\n")
