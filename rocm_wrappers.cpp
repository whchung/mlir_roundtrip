#include <rocblas.h>
#include <miopen/miopen.h>
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

void rocblas_sgemm(float *da, float *db, float *dc, int m, int n, int k, int lda, int ldb, int ldc) {
  rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
  float alpha = 1.0f, beta = 1.0f;

  rocblas_handle handle;
  rocblas_create_handle(&handle);
  rocblas_sgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc);
  rocblas_destroy_handle(handle);
}

void miopen_conv2d(float *filter, float *input, float *output,
                   int n, int c, int hi, int wi, int k, int y, int x, int ho, int wo,
                   int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
  miopenHandle_t handle;
  miopenTensorDescriptor_t inputDesc, outputDesc, filterDesc;
  miopenConvolutionDescriptor_t convDesc;
  miopenCreate(&handle);
  miopenCreateConvolutionDescriptor(&convDesc);
  miopenInitConvolutionDescriptor(convDesc,
                                  /* c_mode */ miopenConvolution,
                                  /* pad_h */ pad_h,
                                  /* pad_w */ pad_w,
                                  /* stride_h */ stride_h,
                                  /* stride_w */ stride_w,
                                  /* dilation_h */ dilation_h,
                                  /* dilation_w */ dilation_w);
  miopenCreateTensorDescriptor(&inputDesc);
  miopenSet4dTensorDescriptor(inputDesc, miopenFloat, n, c, hi, wi);
  miopenSet4dTensorDescriptorLayout(inputDesc, "nchw");

  miopenCreateTensorDescriptor(&outputDesc);
  miopenSet4dTensorDescriptor(outputDesc, miopenFloat, n, k, ho, wo);
  miopenSet4dTensorDescriptorLayout(outputDesc, "nkhw");

  miopenCreateTensorDescriptor(&filterDesc);
  miopenSet4dTensorDescriptor(filterDesc, miopenFloat, k, c, y, x);
  miopenSet4dTensorDescriptorLayout(filterDesc, "kcyx");

  size_t workSpaceSize = 0;
  miopenConvolutionForwardGetWorkSpaceSize(handle,
                                           /* wDesc */ filterDesc,
                                           /* xDesc */ inputDesc,
                                           /* convDesc */ convDesc,
                                           /* yDesc */ outputDesc,
                                           /* workSpaceSize */ &workSpaceSize);

  void *workSpace = nullptr;
  if (workSpaceSize)
    hipMalloc(&workSpace, workSpaceSize);

  int algoCount = 0;
  miopenConvAlgoPerf_t perfResult;
  miopenFindConvolutionForwardAlgorithm(handle,
                                        /* xDesc */ inputDesc,
                                        /* x */ input,
                                        /* wDesc */ filterDesc,
                                        /* w */ filter,
                                        /* convDesc */ convDesc,
                                        /* yDesc */ outputDesc,
                                        /* y */ output,
                                        /* requestAlgoCount */ 1,
                                        /* returnedAlgoCount */ &algoCount,
                                        /* perfResults */ &perfResult,
                                        /* workSpace */ workSpace,
                                        /* workSpaceSize */ workSpaceSize,
                                        /* exhaustiveSearch */ false);

  float alpha = 1.0;
  float beta = 0.0;
  miopenConvolutionForward(handle,
                           /* alpha */ &alpha,
                           /* xDesc */ inputDesc,
                           /* x */ input,
                           /* wDesc */ filterDesc,
                           /* w */ filter,
                           /* convDesc */ convDesc,
                           /* algo */ perfResult.fwd_algo,
                           /* beta */ &beta,
                           /* yDesc */ outputDesc,
                           /* y */ output,
                           /* workSpace */ workSpace,
                           /* workSpaceSize */ workSpaceSize); 

  if (workSpaceSize > 0)
    hipFree(workSpace);
  miopenDestroy(handle);
}
