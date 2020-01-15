#include <mlir_runner_utils.h>

#include <cstdint>
#include <iostream>
#include <map>

void hip_alloc(void**, int);
void hip_free(void*);
void hip_fill(void*, float, int);
void hip_vecadd(void*, void*, void*, int);
void hip_memcpydtoh(void*, void*, int);

void rocblas_sgemm(float*, float*, float*, int, int, int, int, int, int);

typedef std::map<void *, void *> HostDeviceMemoryMap;
static HostDeviceMemoryMap g_mmap;
 
extern "C" void gpu_alloc(StridedMemRefType<char, 1> *memref_host) {
  void *gpu_ptr = nullptr;
  hip_alloc(&gpu_ptr, memref_host->sizes[0]);
  g_mmap[memref_host->data] = gpu_ptr;
}

extern "C" void gpu_dealloc(StridedMemRefType<char, 1> *memref_host) {
  if (g_mmap.find(memref_host->data) != g_mmap.end()) {
    hip_free(memref_host->data);
  } else {
    std::cout << "g_mmap doesn't contain specified host address!\n";
  }
}

extern "C" float gpu_load(StridedMemRefType<float, 1> *memref_host, int64_t index) {
  float result = 0.f;

  if (g_mmap.find(memref_host->data) != g_mmap.end()) {
    hip_memcpydtoh(memref_host->data, g_mmap[memref_host->data], memref_host->sizes[0] * sizeof(float));
    result = memref_host->data[index];
  } else {
    std::cerr << "g_mmap doesn't contain specified host address!\n";
  }
  return result;
}

extern "C" float gpu_load2d(StridedMemRefType<float, 2> *memref_host, int64_t y, int64_t x) {
  float result = 0.f;

  if (g_mmap.find(memref_host->data) != g_mmap.end()) {
    hip_memcpydtoh(memref_host->data, g_mmap[memref_host->data], memref_host->sizes[0] * memref_host->sizes[1] * sizeof(float));

    result = memref_host->data[y * memref_host->strides[0] + x];
  } else {
    std::cerr << "g_mmap doesn't contain specified host address!\n";
  }

  return result;
}

extern "C" void linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X,
                                          float f) {
  // Fill CPU memref.
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    *(X->data + X->offset + i * X->strides[0]) = f;

  // Fill GPU memref.
  if (g_mmap.find(X->data) != g_mmap.end()) {
    hip_fill(g_mmap[X->data], f, X->sizes[0]);
  }
}

__attribute__((always_inline)) extern "C" void
external_func(
    StridedMemRefType<float, 1> *A, StridedMemRefType<float, 1> *B,
    StridedMemRefType<float, 1> *C) {
  if ((g_mmap.find(A->data) != g_mmap.end()) &&
      (g_mmap.find(B->data) != g_mmap.end()) &&
      (g_mmap.find(C->data) != g_mmap.end())) {
    hip_vecadd(g_mmap[A->data], g_mmap[B->data], g_mmap[C->data],
               A->sizes[0]);
  }
}

__attribute__((always_inline)) extern "C" void
linalg_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32(
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    StridedMemRefType<float, 2> *C) {
  // GEMM on GPU.
  if ((g_mmap.find(A->data) != g_mmap.end()) &&
      (g_mmap.find(B->data) != g_mmap.end()) &&
      (g_mmap.find(C->data) != g_mmap.end())) {
    rocblas_sgemm(static_cast<float*>(g_mmap[A->data]),
                  static_cast<float*>(g_mmap[B->data]),
                  static_cast<float*>(g_mmap[C->data]),
                  C->sizes[0], C->sizes[1], A->sizes[1],
                  A->strides[0], B->strides[0], C->strides[0]);
  } else {
    std::cerr << "g_mmap doesn't contain specified host address!\n";
  }
}
