
#strided1D = (d0) -> (d0)
#strided2D = (d0, d1)[s0] -> (d0 * s0 + d1)

// allocates and returns a 1D memref of size %s filled with value %f.
func @alloc_filled_f32(%s: index, %f: f32) -> memref<?xi8> {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c4 = constant 4: index
  %s4 = muli %s, %c4: index
  %buf = alloc(%s4) {alignment = 256} : memref<?xi8>
  %fp32_view = view %buf[%s][] : memref<?xi8> to memref<?xf32, #strided1D>
  linalg.fill(%fp32_view, %f) : memref<?xf32, #strided1D>, f32
  return %buf : memref<?xi8>
}

func @matmul() -> f32 {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c6 = constant 6: index
  %c7 = constant 7: index

  %m = constant 1024: index
  %k = constant 1024: index
  %n = constant 1024: index

  %mk = constant 1048576: index
  %kn = constant 1048576: index
  %mn = constant 1048576: index

  %f1 = constant 1.0e+0 : f32
  %f2 = constant 2.0e+0 : f32
  %f10 = constant 10.0e+0 : f32

  // allocate and fill memrefs
  %bA = call @alloc_filled_f32(%mk, %f2) : (index, f32) -> memref<?xi8>
  %bB = call @alloc_filled_f32(%kn, %f1) : (index, f32) -> memref<?xi8>
  %bC = call @alloc_filled_f32(%mn, %f10) : (index, f32) -> memref<?xi8>

  %A = view %bA[][%m, %k] : memref<?xi8> to memref<?x?xf32, #strided2D>
  %B = view %bB[][%k, %n] : memref<?xi8> to memref<?x?xf32, #strided2D>
  %C = view %bC[][%m, %n] : memref<?xi8> to memref<?x?xf32, #strided2D>

  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>,
                              memref<?x?xf32, #strided2D>,
                              memref<?x?xf32, #strided2D>
  %res = load %C[%c6, %c7] : memref<?x?xf32, #strided2D>

  dealloc %bC : memref<?xi8>
  dealloc %bB : memref<?xi8>
  dealloc %bA : memref<?xi8>

  return %res : f32
}
