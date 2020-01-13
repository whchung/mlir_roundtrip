#include <hip/hip_runtime.h>
#include <iostream>

__attribute__((visibility("default"))) void hip_alloc(void **ptr, int size) {
  hipMalloc(ptr, size);
}

__attribute__((visibility("default"))) void hip_free(void *ptr) {
  hipFree(ptr);
}

__global__ void FillKernel(float *ptr, float v, size_t N) {
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  size_t stride = hipBlockDim_x * hipGridDim_x;

  for (size_t i = offset; i < N; i += stride) {
    ptr[i] = v;
  }
}

void hip_fill(void *ptr, float v, int size) {
  hipLaunchKernelGGL(FillKernel, dim3(1), dim3(256), 0, 0, static_cast<float*>(ptr), v, static_cast<size_t>(size));
}

__global__ void VecAddKernel(float *A, float *B, float *C, size_t N) {
  size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  size_t stride = hipBlockDim_x * hipGridDim_x;

  for (size_t i = offset; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

void hip_vecadd(void *A, void *B, void *C, int size) {
  hipLaunchKernelGGL(VecAddKernel, dim3(1), dim3(256), 0, 0, static_cast<float*>(A), static_cast<float*>(B), static_cast<float*>(C), static_cast<size_t>(size));
}

void hip_memcpydtoh(void *host, void *device, int bytes) {
  hipMemcpy(host, device, bytes, hipMemcpyDeviceToHost);
}
