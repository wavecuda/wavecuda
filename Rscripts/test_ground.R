library(wavecuda)
library(wavethresh)

x1 <- rnorm(1024)
x2 <- rnorm(16)
x3 <- 1:16

w1 <- GPUTransform(x1,"FWD",0,"DWT","Haar")
w2 <- GPUTransform(x2,"FWD",0,"DWT","Haar")
w3 <- GPUTransform(x3,"FWD",0,"DWT","Haar")

w3_df <- WST.to.DT(w3)
nas3 <- wd(x3,filter.number=1,family="DaubExPhase")
all.equal(nw3$D, nas3$D)
all.equal(nw3$C, nas3$C)
## not equal - haven't done scaling coeffs yet...
