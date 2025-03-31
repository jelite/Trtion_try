#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_add_kernel_caba951e_0d1d2d34(void);
void load_add_kernel_caba951e_0d1d2d34(void);
// tt-linker: add_kernel_caba951e_0d1d2d34:CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements:1024_warps4xstages3
CUresult add_kernel_caba951e_0d1d2d34(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);