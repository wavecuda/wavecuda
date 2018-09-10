library(wavecuda)
library(wavethresh)

x1 <- rnorm(1024)
x2 <- rnorm(16)
x3 <- 1:16

w1 <- GPUTransform(x1,"FWD",0,"DWT","Haar")
w2 <- GPUTransform(x2,"FWD",0,"DWT","Haar")
w3 <- GPUTransform(x3,"FWD",0,"DWT","Haar")
w32 <- GPUTransform(x3,"FWD",2,"DWT","Haar")
w4 <- GPUTransform(x3,"FWD",0,"MODWT","Haar")

w3_df <- WST.to.DT(w3)
nw3 <- WST.to.wavethresh(w3)
nas3 <- wd(x3,filter.number=1,family="DaubExPhase")
all.equal(nw3$D, nas3$D)
all.equal(nw3$C, nas3$C)
## not equal - haven't done scaling coeffs yet...
## and will only do top ones

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
