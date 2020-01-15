#include <cblas.h>
#include <mlir_runner_utils.h>

extern "C" void linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X,
                                          float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    *(X->data + X->offset + i * X->strides[0]) = f;
}

__attribute__((always_inline)) extern "C" void
external_func(
    StridedMemRefType<float, 1> *A, StridedMemRefType<float, 1> *B,
    StridedMemRefType<float, 1> *C) {
   
  //std::cout << std::endl;
  //printMemRefMetaData(std::cerr, *A);
  //std::cout << std::endl;
  //printMemRefMetaData(std::cerr, *B);
  //std::cout << std::endl;
  //printMemRefMetaData(std::cerr, *C);
  //std::cout << std::endl;

  for (int64_t i = 0; i < A->sizes[0]; ++i) {
    C->data[i] = A->data[i] + B->data[i];
  }
}
