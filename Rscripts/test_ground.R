library(wavecuda)
library(wavethresh)
library(wmtsa)

x1 <- rnorm(1024)
x2 <- rnorm(16)
x3 <- 1:16
x4 <- wmtsa::make.signal("doppler", n = 1024, snr = 5)
x4 <- x4@data

w1 <- GPUTransform(x1,"FWD",0,"DWT","Haar")
w2 <- GPUTransform(x2,"FWD",0,"DWT","Haar")
w3 <- GPUTransform(x3,"FWD",0,"DWT","Haar")
w32 <- GPUTransform(x3,"FWD",2,"DWT","Haar")
w33 <- GPUTransform(x3,"FWD",3,"DWT","Haar")
w4 <- GPUTransform(x3,"FWD",0,"MODWT","Haar")
w5 <- GPUTransform(x4, "FWD", 0, "DWT", "Haar")
w6 <- GPUTransform(x4, "FWD", 0, "MODWT", "Haar")

plot(w5)
plot(w6)

w3_df <- WSTtoDT(w3)
nw3 <- WSTtoWavethresh(w3)
nw5 <- WSTtoWavethresh(w5)
nw6 <- WSTtoWavethresh(w6)
nas3 <- wd(x3,filter.number=1,family="DaubExPhase")
all.equal(nw3$D, nas3$D)
all.equal(nw3$C, nas3$C)
## not equal - haven't done scaling coeffs yet...
## and will only do top ones
nas5 <- wd(x4, filter.number=1,family="DaubExPhase")
plot(nas5)
plot(w5)
## identical

w3t <- CPUThreshold(w3, thresh = 1, hard.soft = "hard", min.level = 1, max.level = 2)

w3ts <- CPUThreshold(w3, thresh = 0.5, hard.soft = "soft", min.level = 1, max.level = 2)

w3t2 <- CPUThreshold(w3, thresh = 3, hard.soft = "hard", min.level = 1, max.level = 4)

x3r <- CPUReconstruct(w3t)

## ####################
## To do:
##
## - check CV code
## - add D4, C6 & LA8 to transform.cpp & transformcuda.cu
## - change x/xmod in WST to xin and w - done
## - propagate that to everything! - done
## - finish wst to dt, complete with scaling coeffs
## - finish wst to wavethresh
## - write GPU, CPU Reconstruct with WST object - done
## - re-write CPU Threshold with WST object - done
## - with check.thresh.inputs - done
## - finish documentation, adding reconstruct - done
## - check that wavecuda_demo in thesis2/figures works
