#ifndef CUDAHEADS_H
#define CUDAHEADS_H

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define HTD cudaMemcpyHostToDevice
#define DTH cudaMemcpyDeviceToHost
#define DTD cudaMemcpyDeviceToDevice
#define HTH cudaMemcpyHostToHost

#endif //ifndef
