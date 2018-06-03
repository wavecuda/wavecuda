library(wavecuda)

library(wavethresh)
library(wmtsa)
library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)



x <- rnorm(16)
ttype <- "DWT"
filt <- "Haar"
filtlen <- 2
transformed <- TRUE
levels <- 4
len <- 16
xmod <- NULL


i <- 6
levels <- i-3
x <- make.signal("doppler",n = 2^i,snr = 10)@data
Xwav <- GPUTransform(x,"FWD",levels,"DWT","Haar")

levelList <- sapply(X = 1:levels, FUN = function(l) getCoeffLevel(Xwav,l,"d"))

## compare with...
xw <- wd(x,filter.number = 1, family="DaubExPhase")
plot(xw, scaling = "by.level")

plotDWT(Xwav)

system.time(p <- plotDWT(Xwav, "dp"))
p

system.time(p <- plotDWT(Xwav, "dt"))
p

Xwav2 <- GPUTransform(x, "FWD", levels, "MODWT.TO", "Haar")
xw2 <- wd(x,filter.number = 1, family="DaubExPhase", type = "station")
plot(xw2, scaling = "by.level")

plotMODWT(Xwav2)

## my plotting is slow
## my modwt plot is shifted!

lineprof(plotMODWT(Xwav2))
