
#strided1D = (d0) -> (d0)

// allocates and returns a 1D memref of size %s filled with value %f.
func @alloc_filled_f32(%s: index, %f: f32) -> memref<?xi8> {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c4 = constant 4: index
  %s4 = muli %s, %c4: index

  %buf = alloc(%s4) {alignment = 256} : memref<?xi8>
  call @gpu_alloc(%buf) : (memref<?xi8>) -> ()

  %fp32_view = view %buf[%s][] : memref<?xi8> to memref<?xf32, #strided1D>
  linalg.fill(%fp32_view, %f) : memref<?xf32, #strided1D>, f32
  return %buf : memref<?xi8>
}

func @conv() -> f32 {
  %c0 = constant 0: index
  %c6 = constant 1: index
  %c7 = constant 1: index

  %n = constant 1 : index
  %c = constant 1 : index
  %h = constant 32 : index
  %w = constant 32 : index

  %k = constant 1 : index
  %y = constant 3 : index
  %x = constant 3 : index

  %ho = constant 30 : index
  %wo = constant 30 : index

  %filter_dim = constant 9 : index
  %input_dim = constant 1024 : index
  %output_dim = constant 900 : index

  %f0 = constant 0.0e+0 : f32
  %f1 = constant 1.0e+0 : f32
  %f2 = constant 2.0e+0 : f32

  // allocate and fill memrefs
  %bFilter = call @alloc_filled_f32(%filter_dim, %f1) : (index, f32) -> memref<?xi8>
  %bInput  = call @alloc_filled_f32(%input_dim, %f2) : (index, f32) -> memref<?xi8>
  %bOutput = call @alloc_filled_f32(%output_dim, %f0) : (index, f32) -> memref<?xi8>

  %filter = view %bFilter[][%k, %c, %y, %x] : memref<?xi8> to memref<?x?x?x?xf32>
  %input  = view %bInput [][%n, %c, %h, %w] : memref<?xi8> to memref<?x?x?x?xf32>
  %output = view %bOutput[][%n, %c, %ho, %wo] : memref<?xi8> to memref<?x?x?x?xf32>

  linalg.conv(%filter, %input, %output) {
    dilations = [1, 1],
    strides = [1, 1]
  } : memref<?x?x?x?xf32>,
      memref<?x?x?x?xf32>,
      memref<?x?x?x?xf32>

  // Load from GPU memref.
  %res = call @gpu_load4d(%output, %c0, %c0, %c6, %c7) : (memref<?x?x?x?xf32>, index, index, index, index) -> f32

  call @gpu_dealloc(%bFilter) : (memref<?xi8>) -> ()
  call @gpu_dealloc(%bInput ) : (memref<?xi8>) -> ()
  call @gpu_dealloc(%bOutput) : (memref<?xi8>) -> ()

  dealloc %bFilter : memref<?xi8>
  dealloc %bInput  : memref<?xi8>
  dealloc %bOutput : memref<?xi8>

  return %res : f32
}

func @gpu_alloc(memref<?xi8>)
func @gpu_dealloc(memref<?xi8>)
func @gpu_load4d(memref<?x?x?x?xf32>, index, index, index, index) -> f32
