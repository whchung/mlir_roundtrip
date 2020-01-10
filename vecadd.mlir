// allocates and returns a 1D memref of size %s filled with value %f.
func @alloc_filled_f32(%s: index, %f: f32) -> memref<?xi8> {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c4 = constant 4: index
  %s4 = muli %s, %c4: index
  %buf = alloc(%s4) {alignment = 256} : memref<?xi8>
  %fp32_view = view %buf[%s][] : memref<?xi8> to memref<?xf32>
  linalg.fill(%fp32_view, %f) : memref<?xf32>, f32
  return %buf : memref<?xi8>
}

#access = [
  (m) -> (m)
]
#trait = {
  args_in = 2, args_out = 1,
  iterator_types = ["parallel"],
  indexing_maps = #access,
  library_call = "external_func"
}

func @vecadd() -> f32 {
  %c0 = constant 0 : index
  %size = constant 1024 : index

  %f0 = constant 0.0e+0 : f32
  %f1 = constant 1.0e+0 : f32
  %f2 = constant 2.0e+0 : f32

  // allocate
  %bA = call @alloc_filled_f32(%size, %f1) : (index, f32) -> memref<?xi8>
  %bB = call @alloc_filled_f32(%size, %f2) : (index, f32) -> memref<?xi8>
  %bC = call @alloc_filled_f32(%size, %f0) : (index, f32) -> memref<?xi8>

  // convert to 1D f32 memref
  %A = view %bA[][%size] : memref<?xi8> to memref<?xf32>
  %B = view %bB[][%size] : memref<?xi8> to memref<?xf32>
  %C = view %bC[][%size] : memref<?xi8> to memref<?xf32>

  linalg.generic #trait %A, %B, %C {
    ^bb(%a: f32, %b: f32, %c: f32):
      %d = addf %a, %b : f32
      linalg.yield %d : f32
  } : memref<?xf32>, memref<?xf32>, memref<?xf32>

  %res = load %C[%c0] : memref<?xf32>

  dealloc %bC : memref<?xi8>
  dealloc %bB : memref<?xi8>
  dealloc %bA : memref<?xi8>

  return %res : f32
}
