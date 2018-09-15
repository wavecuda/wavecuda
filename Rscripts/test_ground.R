library(wavecuda)
library(wavethresh)
library(wmtsa)

x1 <- rnorm(1024)
x2 <- rnorm(16)
x3 <- 1:16
x4 <- wmtsa::make.signal("doppler", n = 1024, snr = 5)
x4 <- x4@data

w1 <- GPUTransform(x1,"FWD",0,"DWT","D4")
w1 <- GPUTransform(x1,"FWD",0,"DWT","Haar")
w2 <- GPUTransform(x2,"FWD",0,"DWT","Haar")
w3 <- GPUTransform(x3,"FWD",0,"DWT","Haar")
w32 <- GPUTransform(x3,"FWD",2,"DWT","Haar")
w33 <- GPUTransform(x3,"FWD",3,"DWT","Haar")
w4 <- GPUTransform(x3,"FWD",0,"MODWT","Haar")
w5 <- GPUTransform(x4, "FWD", 0, "DWT", "Haar")
w6 <- GPUTransform(x4, "FWD", 0, "MODWT", "Haar")
w7 <- GPUTransform(x3,"FWD",0,"DWT","D4")
w8 <- GPUTransform(x3,"FWD",0,"DWT","C6")
w9 <- GPUTransform(x3,"FWD",0,"DWT","LA8")
w10 <- CPUTransform(x3,"FWD",0,"DWT","D4") 
w11 <- CPUTransform(x3,"FWD",0,"DWT","C6") 
w12 <- CPUTransform(x3,"FWD",0,"DWT","LA8")


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

## ####################
## Checking wavecuda functions...

library(dplyr)
library(purrr)

x <- wmtsa::make.signal("doppler", n = 1024, snr = Inf)
x <- x@data

filters <- c("Haar", "D4", "C6", "LA8")
ttypes <- c("DWT", "MODWT")
## funcs <- c(list(CPUTransform),list(GPUTransform))

## might do this later with purrr, but for now stick to a plain old loop

## for(filt in filters){
##     for(tt in ttypes){

safeCPUTransform <- safely(CPUTransform)
safeGPUTransform <- safely(GPUTransform)

optTab <- expand.grid(Filter = filters, Ttype = ttypes, stringsAsFactors = FALSE) %>%
    as.tbl()

cpuTab <- optTab %>%
    mutate(CPUT = map2(.x = Ttype, .y = Filter, .f = function(X,Y){safeCPUTransform(x, direction = "FWD", nlevels = 0, transform.type = X, filter = Y)})) %>%
    mutate(CPUerror = map(.x = CPUT, .f = function(X) X$error))

cpuTab %>%
    select(Filter, Ttype, CPUerror)
        

gpuTab <- optTab %>%
    mutate(GPUT = map2(.x = Ttype, .y = Filter, .f = function(X,Y){safeGPUTransform(x, direction = "FWD", nlevels = 0, transform.type = X, filter = Y)})) %>%
    mutate(GPUerror = map(.x = GPUT, .f = function(X) X$error))

gpuTab %>%
    select(Filter, Ttype, GPUerror)

plot(GPUTransform(x, "FWD", 0, "DWT", "LA8"))
plot(GPUTransform(x, "FWD", 0, "DWT", "C6"))
plot(GPUTransform(x, "FWD", 0, "DWT", "D4"))
plot(GPUTransform(x, "FWD", 0, "DWT", "Haar"))

plot(CPUTransform(x, "FWD", 0, "DWT", "LA8")) ## looks shifted
plot(CPUTransform(x, "FWD", 0, "DWT", "C6")) ## looks shifted
plot(CPUTransform(x, "FWD", 0, "DWT", "D4"))
plot(CPUTransform(x, "FWD", 0, "DWT", "Haar"))

plot(x, type = "l")

plot(CPUSmooth(x, nlevels = 10, transform.type = "DWT", filter = "Haar", thresh.type = "univ", hard.soft = "hard", min.level = 1, max.level = 10),type = "l")

plot(CPUReconstruct(CPUTransform(x, "FWD", 0, "DWT", "D4")))

xn <- wmtsa::make.signal("doppler", n = 1024, snr = 5)
xn <- xn@data

xSmoothed <- GPUSmooth(xn, nlevels = 10, transform.type = "DWT", filter = "Haar", thresh.type = "cv", hard.soft = "hard", min.level = 1, max.level = 6) ## seg faults

xSmoothed <- GPUSmooth(xn, nlevels = 8, transform.type = "DWT", filter = "C6", thresh.type = "cv", hard.soft = "hard", min.level = 1, max.level = 8)

xSmoothed <- CPUSmooth(xn, nlevels = 10, transform.type = "DWT", filter = "Haar", thresh.type = "cv", hard.soft = "hard", min.level = 3, max.level = 8) ## doesn't seg fault!
