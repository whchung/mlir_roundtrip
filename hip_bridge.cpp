#include <mlir_runner_utils.h>

#include <cstdint>
#include <iostream>
#include <map>

void hip_alloc(void**, int);
void hip_free(void*);
void hip_fill(void*, float, int);
void hip_vecadd(void*, void*, void*, int);
void hip_memcpydtoh(void*, void*, int);

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
    hip_memcpydtoh(memref_host->data, g_mmap[memref_host->data], memref_host->sizes[0]);
    result = memref_host->data[index];
  } else {
    std::cerr << "g_mmap doesn't contain specified host address!\n";
  }
  return result;
}

extern "C" void linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X,
                                          float f) {
  if (g_mmap.find(X->data) != g_mmap.end()) {
    hip_fill(g_mmap[X->data], f, X->sizes[0] / sizeof(float));
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
               A->sizes[0] / sizeof(float));
  }
}

#if 0
int main() {
  char* p = nullptr;

  std::cout << "call hipMalloc\n";
  std::cout << p << "\n";

  hip_alloc(&p, 1024);

  std::cout << "after hipMalloc:\n";
  std::cout << p << "\n";

  std::cout << "call hipFree\n";
  hip_free(p);

  return 0;
}
#endif
