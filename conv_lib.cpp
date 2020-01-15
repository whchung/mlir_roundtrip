#include <cblas.h>
#include <mlir_runner_utils.h>

extern "C" void linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X,
                                          float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    *(X->data + X->offset + i * X->strides[0]) = f;
}

void printMemRef(StridedMemRefType<float, 4> *memref) {
  for (int y = 0; y < memref->sizes[2]; y++) {
    for (int x = 0; x <memref->sizes[3]; x++) {
      std::cout <<memref->data[y * memref->strides[2] + x * memref->strides[3]] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

__attribute__((always_inline)) extern "C" void
linalg_conv_viewsxsxsxsxf32_viewsxsxsxsxf32_viewsxsxsxsxf32(
    StridedMemRefType<float, 4> *filter,
    StridedMemRefType<float, 4> *input,
    StridedMemRefType<float, 4> *output) {

  //std::cout << std::endl;
  //printMemRefMetaData(std::cerr, *filter);
  //std::cout << std::endl;
  //printMemRefMetaData(std::cerr, *input);
  //std::cout << std::endl;
  //printMemRefMetaData(std::cerr, *output);
  //std::cout << std::endl;

  //std::cout << "Filter:\n";
  //printMemRef(filter);
  //std::cout << "Input:\n";
  //printMemRef(input);
  //std::cout << "Output:\n";
  //printMemRef(output);

  // convolution on CPU.
  int io_y_offset = (filter->sizes[2] / 2);
  int io_x_offset = (filter->sizes[3] / 2);

  for (int n = 0; n < output->sizes[0]; ++n) {
    for (int k = 0; k < output->sizes[1]; ++k) {
      for (int c = 0; c < input->sizes[1]; ++c) {

        for (int oy = 0; oy < output->sizes[2]; ++oy) {
          for (int ox = 0; ox < output->sizes[3]; ++ox) {

            float o = 0.0f;

            for (int fy = 0; fy < filter->sizes[2]; ++fy) {
              for (int fx = 0; fx < filter->sizes[3]; ++fx) {
                int iy = oy + io_y_offset + (fy - (filter->sizes[2] / 2));
                int ix = ox + io_x_offset + (fx - (filter->sizes[3] / 2));

                o += filter->data[k * filter->strides[0] +
                                  c * filter->strides[1] +
                                  fy * filter->strides[2] +
                                  fx * filter->strides[3]]
                     *
                     input->data[n * input->strides[0] +
                                 c * input->strides[1] +
                                 iy * input->strides[2] +
                                 ix * input->strides[3]];

              }
            }

            output->data[n * output->strides[0] +
                         k * output->strides[1] +
                         oy * output->strides[2] +
                         ox * output->strides[3]] = o;

          }
        }

      }
    }
  }
}
