# WaveCuda: an R package for CUDA-Accelerated Wavelet Transforms

We have written CUDA-accelerated wavelet transforms for the wavelet filters Haar, D4, C6 and LA8 using the Lifting Scheme.

An NVIDIA CUDA-enabled GPU is required for running the GPU transforms, and the CUDA sdk is required for compiling the library. Download at www.nvidia.com/getcuda

## Installation instructions

- Clone the repository
- Navigate to the src directory, then compile with `make`.
- Navigate to the wavecuda package directory. Install the package with:
-- devtools::build()
-- devtools::install()

## Basic demonstration of functionality

```r
library(wavecuda)

set.seed(10)
x <- wmtsa::make.signal("doppler", n = 1024, snr = 5)
## requires wmtsa package
x <- x@data

w1 <- CPUTransform(x, direction = "FWD", nlevels = 0, transform.type = "DWT",
                   filter = "D4")

w2 <- GPUTransform(x, direction = "FWD", nlevels = 0, transform.type = "DWT",
                   filter = "C6")

w3 <- GPUTransform(x, direction = "FWD", nlevels = 0, transform.type = "MODWT",
                   filter = "Haar")

xReconstructed <- GPUReconstruct(w3)

sSmoothed <- GPUSmooth(x, nlevels = 10, transform.type = "MODWT", filter = "Haar", thresh.type = "cv", hard.soft = "soft", min.level = 1, max.level = 10)
```

