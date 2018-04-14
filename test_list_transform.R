source("rfunctions.R")
library(wmtsa)

n1 <- 32
x1 <- make.signal("doppler",n1)@data

n2 <- 64
x2 <- make.signal("doppler",n2)@data

res.list <- GPUTransformList(list(x1,x2),rep("FWD",2),rep(3,2),rep("DWT",2),rep("Haar",2))

## so it is possible! But we ought to protect things, & fully implement
