High-level op
=============

```mlir
linalg.conv -> miopen.conv2d

miopen.conv2d(%filter, %input, %output) {
    filter_layout = ["k", "c", "y", "x"],
    input_layout = ["n", "c", "hi", "wi"],
    output_layout = ["n", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
} : memref<?x?x?x?xf32>,
    memref<?x?x?x?xf32>,
    memref<?x?x?x?xf32>
```
```mlir
miopen.conv2d_bwd_data(%filter, %input, %output) {
    filter_layout = ["k", "c", "y", "x"],
    input_layout = ["n", "c", "hi", "wi"],
    output_layout = ["n", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
} : memref<?x?x?x?xf32>,
    memref<?x?x?x?xf32>,
    memref<?x?x?x?xf32>
```

-------------------------------------------------------------------------------

Conv Forward Transform
===============

An example based on NCHW/KCYX/NKHW:

```mlir
// filter tensor
%filter_gemmK_gemmM = miopen.transform(%filter) {
  gridwise_gemm_argument_position = 0
  layout = [
    {
      dimensions = [0],
      names = ["gemmK"],
      transformation = "merge",
      source_dimensions = [1, 2, 3],
      source_names = ["c", "y", "x"]
    },
    {
      dimensions = [1],
      names = ["gemmM"],
      transformation = "passthrough",
      source_dimensions = [0],
      source_names = ["n"]
    }
  ],
  output_layout = ["gemmK", "gemmM"],
  source_layout = ["k", "c", "y", "x"]
} : memref<?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// input tensor
%input_n_c_hipad_wipad = miopen.transform(%input) {
  layout = [
    {
      dimensions = [0],
      names = ["n"],
      transformation = "passthorugh",
      source_dimensions = [0],
      source_names = ["n"]
    },
    {
      dimensions = [1],
      names = ["c"],
      transformation = "passthorugh",
      source_dimensions = [1],
      source_names = ["c"]
    },
    {
      dimensions = [2],
      names = ["hipad"],
      transformation = "pad",
      parameters = [0, 0],
      source_dimensions = [2],
      source_names = ["hi"]
    },
    {
      dimensions = [3],
      names = ["wipad"],
      transformation = "pad",
      parameters = [0, 0],
      source_dimensions = [3],
      source_names = ["wi"]
    }
  ],
  output_layout = ["n", "c", "hipad", "wipad"],
  source_layout = ["n", "c", "h", "w"]
} : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>

%input_n_c_y_ho_x_wo = miopen.transform(%input_n_c_hipad_wipad) {
  layout = [
    layout = [
      {
        dimensions = [0],
        names = ["n"],
        transformation = "passthrough",
        source_dimensions = [0],
        source_names = ["n"]
      },
      {
        dimensions = [1],
        names = ["c"],
        transformation = "passthrough",
        source_dimensions = [1],
        source_names = ["c"]
      },
      {
        dimensions = [2, 3],
        names = ["y", "ho"],
        transformation = "embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [2],
        source_names = ["hipad"]
      },
      {
        dimensions = [4, 5],
        names = ["x", "wo"],
        transformation = "embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [2],
        source_names = ["wipad"]
      }
  ],
  intermediate_layout = ["n", "c", "hipad", "wipad"],
  output_layout = ["n", "c", "y", "ho", "x", "wo"]
} : memref<?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32>

%input_gemmK_gemmN = miopen.transform(%input_n_c_y_ho_x_wo) {
  gridwise_gemm_argument_position = 1
  layout = [
    {
      dimensions = [0],
      names = ["gemmK"],
      transformation = "merge",
      source_dimensions = [1, 2, 4],
      source_names = ["c", "y", "x"]
    },
    {
      dimensions = [1],
      names = ["gemmN"],
      transformation = "merge",
      source_dimensions = [0, 3, 5],
      source_names = ["n", "ho", "wo"]
    }
  ],
  intermediate_layout = ["n", "c", "y", "ho", "x", "wo"],
  output_layout = ["gemmK", "gemmN"]
} : memref<?x?x?x?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// output tensor
%output_gemmM_gemmN = miopen.transform(%output) {
  gridwise_gemm_argument_position = 2
  layout = [
    {
      dimensions = [0],
      names = ["gemmM"],
      transformation = "passthrough",
      source_dimensions = [1],
      source_names = ["k"]
    },
    {
      dimensions = [1],
      names = ["gemmN"],
      transformation = "merge",
      source_dimensions = [0, 2, 3],
      source_names = ["n", "ho", "wo"]
    }
  ],
  output_layout = ["gemmM", "gemmN"],
  source_layout = ["n", "ko", "ho", "wo"]
} : memref<?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// apply gridwise GEMM
miopen.gridwise_gemm(%filter_gemmK_gemmM, %input_gemmK_gemmN, %output_gemmM_gemmN) {
  kernel_algorithm = "v4r4",
  filter_dimension = [?, ?, ?, ?],
  filter_layout = ["k", "c", "y", "x"],
  input_dimension = [?, ?, ?, ?],
  input_layout = ["n", "c", "hi", "wi"],
  output_dimension = [?, ?, ?, ?],
  output_layout = ["n", "k", "ho", "wo"],
  dilations = [1, 1],
  strides = [1, 1],
  padding = [[0, 0], [0, 0]]
} : memref<?x?xf32>,
    memref<?x?xf32>,
    memref<?x?xf32>
```
-------------------------------------------------------------------------------

Conv Backward Data Transform
===============

An example based on NCHW/KCYX/NKHW:

```mlir
// filter tensor
%filter_gemmK_gemmM = miopen.transform(%filter) {
  gridwise_gemm_argument_position = 0
  layout = [
    {
      dimensions = [0],
      names = ["gemmK"],
      transformation = "passthrough",
      source_dimensions = [0],
      source_names = ["k"]
    },
    {
      dimensions = [1],
      names = ["gemmM"],
      transformation = "merge",
      source_dimensions = [1, 2, 3],
      source_names = ["c", "y", "x"]
    }
  ],
  output_layout = ["gemmK", "gemmM"],
  source_layout = ["k", "c", "y", "x"]
} : memref<?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// output_diff tensor
%output_gemmK_gemmN = miopen.transform(%output_diff) {
  gridwise_gemm_argument_position = 1,
  layout = [
    {
      dimensions = [0],
      names = ["gemmK"],
      transformation = "passthrough",
      source_dimensions = [1],
      source_names = ["k"]
    },
    {
      dimensions = [1],
      names = ["gemmN"],
      transformation = "merge",
      source_dimensions = [0, 2, 3],
      source_names = ["n", "ho", "wo"]
    }
  ],
  output_layout = ["gemmK", "gemmN"],
  source_layout = ["n", "k", "ho", "wo"]
} : memref<?x?x?x?xf32> to memref<?x?xf32>
```


```mlir
// input tensor
%input_n_c_hipad_wipad = miopen.transform(%input) {
  layout = [
    {
      dimensions = [0],
      names = ["n"],
      transformation = "passthorugh",
      source_dimensions = [0],
      source_names = ["n"]
    },
    {
      dimensions = [1],
      names = ["c"],
      transformation = "passthorugh",
      source_dimensions = [1],
      source_names = ["c"]
    },
    {
      dimensions = [2],
      names = ["hipad"],
      transformation = "pad",
      parameters = [0, 0],
      source_dimensions = [2],
      source_names = ["hi"]
    },
    {
      dimensions = [3],
      names = ["wipad"],
      transformation = "pad",
      parameters = [0, 0],
      source_dimensions = [3],
      source_names = ["wi"]
    }
  ],
  output_layout = ["n", "c", "hipad", "wipad"],
  source_layout = ["n", "c", "h", "w"]
} : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>

%input_n_c_y_ho_x_wo = miopen.transform(%input_n_c_hipad_wipad) {
  layout = [
    layout = [
      {
        dimensions = [0],
        names = ["n"],
        transformation = "passthrough",
        source_dimensions = [0],
        source_names = ["n"]
      },
      {
        dimensions = [1],
        names = ["c"],
        transformation = "passthrough",
        source_dimensions = [1],
        source_names = ["c"]
      },
      {
        dimensions = [2, 3],
        names = ["y", "ho"],
        transformation = "embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [2],
        source_names = ["hipad"]
      },
      {
        dimensions = [4, 5],
        names = ["x", "wo"],
        transformation = "embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [2],
        source_names = ["wipad"]
      }
  ],
  intermediate_layout = ["n", "c", "hipad", "wipad"]
  output_layout = ["n", "c", "y", "ho", "x", "wo"]
} : memref<?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32>

%input_gemmM_gemmN = miopen.transform(%input_n_c_y_ho_x_wo) {
  gridwise_gemm_argument_position = 2,
  layout = [
    {
      dimensions = [0],
      names = ["gemmM"],
      transformation = "merge",
      source_dimensions = [1, 2, 4],
      source_names = ["c", "y", "x"]
    },
    {
      dimensions = [1],
      names = ["gemmN"],
      transformation = "merge",
      source_dimensions = [0, 3, 5],
      source_names = ["n", "ho", "wo"]
    }
  ],
  intermediate_layout = ["n", "c", "y", "ho", "x", "wo"],
  output_layout = ["gemmM", "gemmN"]
} : memref<?x?x?x?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// apply gridwise GEMM
miopen.gridwise_gemm(%filter_gemmK_gemmM, %output_gemmK_gemmN, %input_gemmM_gemmN) {
  kernel_algorithm = "backward_data_v1r1",
  filter_dimension = [?, ?, ?, ?],
  filter_layout = ["k", "c", "y", "x"],
  input_dimension = [?, ?, ?, ?],
  input_layout = ["n", "c", "hi", "wi"],
  output_dimension = [?, ?, ?, ?],
  output_layout = ["n", "k", "ho", "wo"],
  dilations = [1, 1],
  strides = [1, 1],
  padding = [[0, 0], [0, 0]]
} : memref<?x?xf32>,
    memref<?x?xf32>,
    memref<?x?xf32>
```

-------------------------------------------------------------------------------

Gridwise GEMM -> Blockwise Slice Copy + Blockwise GEMM + Threadwise Slice Copy

```mlir

miopen.gridwise_gemm_ex(%matrix_a, %matrix_b, %matric_c) {
  block_size = 256,

  m_per_block = 128,
  n_per_block = 128,
  k_per_block = 16,

  m_per_thread = 64,
  n_per_thread = 64,
  k_per_thread = 16,

  m_level0_cluster = 16,
  n_level0_cluster = 16,
  m_level1_cluster = 16,
  n_level1_cluster = 16,

  matrix_a_source_vector_read_dim = 0,
  matrix_a_source_data_per_read = 4,
  matrix_a_dest_data_per_write_dim_m = 4,

  matrix_b_source_vector_read_dim = 1,
  matrix_b_source_data_per_read = 4,
  matrix_b_dest_data_per_write_dim_n = 4,

  matrix_c_source_dest_vector_read_write_dim = 3,
  matrix_c_dest_data_per_write = 1
} : memref<kxmxf32>, memref<kxnxf32>, memref<mxnxf32>

```

```mlir

# %shared_block_size is computed from the following parameters:
# - matrix_a_dest_data_per_write_dim_m
# - matrix_b_dest_data_per_write_dim_n
# - m_per_thread
# - n_per_thread
# - m_per_block
# - n_per_block
# - n, m, k
#
# LDS memory address space is fixed at 3.
%block_shared = miopen.alloc(%shared_block_size, %c3) : memref<?xi8, #3>

# Views for Matrix A on LDS memory

# %block_a is an 1-D subview of %block_shared
%block_a = miopen.subview(%block_shared, %c0) : memref<?xi8, #3> to memref<?xi8, #3>

# %block_a_even is an 1-D subview of %block_a
%block_a_even = miopen.subview(%block_a, %c0) : memref<?xi8, #3> to memref<?xi8, #3>

# %matrix_block_a_even is an 2-D subview of %block_a
%matrix_block_a_even = miopen.subview(%block_a, %c0) { dimension = [%k, %m] } : memref<?xi8, #3> to memref<?x?xf32, #3>

# %block_a_odd is an 1-D subview of %block_a
# %block_a_size is computed similiar with %shared_block_size
%block_a_odd  = miopen.subview(%block_a, %block_a_size) : memref<?xi8, #3> to memref<?xi8, #3>

# %matrix_block_a_odd is an 2-D subview of %block_a
%matrix_block_a_odd = miopen.subview(%block_a_odd, %block_a_size) { dimension = [%k, %m] } : memref<?xi8, #3> to memref<?x?xf32, #3>


# Views for Matrix B on LDS memory

# %block_b is an 1-D subview of %block_shared
%block_b = miopen.subview(%block_shared, %c0) : memref<?xi8, #3> to memref<?xi8, #3>

# %block_b_even is an 1-D subview of %block_b
%block_b_even = miopen.subview(%block_b, %c0) : memref<?xi8, #3> to memref<?xi8, #3>

# %matrix_block_b_even is an 2-D subview of %block_b
%matrix_block_b_even = miopen.subview(%block_b, %c0) { dimension = [%k, %m] } : memref<?xi8, #3> to memref<?x?xf32, #3>

# %block_b_odd is an 1-D subview of %block_b
# %block_b_size is computed similiar with %shared_block_size
%block_b_odd  = miopen.subview(%block_b, %block_b_size) : memref<?xi8, #3> to memref<?xi8, #3>

# %matrix_block_b_odd is an 2-D subview of %block_b
%matrix_block_b_odd = miopen.subview(%block_b_odd, %block_b_size) { dimension = [%k, %m] } : memref<?xi8, #3> to memref<?x?xf32, #3>


# %matrix_c_size is computed from the following formula:
# m_per_block / (m_per_thread * m_level0_cluster * m_level1_cluster) * m_per_thread * n_per_block / (n_per_thread * n_level0_cluster * n_level1_cluster) * n_per_thread
#
# private address space is fixed as a constant 5.
%thread_c = miopen.alloc(%matrix_c_size, %c5) : memref<?xi8, #5>
# %matrix_thread_c is an 2-D subview of %thread_c
%matrix_thread_c = miopen.subview(%thread_c, %c0) { dimension = [%m, %n] } : memref<?xi8, #5> to memref<?x?xf32, #5>

# %blockwise_copy_matrix_a = (k_per_block / A_BLOCK_COPY_CLUSTER_LENGTH_GEMM_K * m_per_block / A_BLOCK_COPY_CLUSTER_LENGTH_GEMM_M
# %blockwise_copy_matrix_b = (k_per_block / B_BLOCK_COPY_CLUSTER_LENGTH_GEMM_K * n_per_block / B_BLOCK_COPY_CLUSTER_LENGTH_GEMM_N
%thread_a_even = miopen.alloc(%blockwise_copy_matrix_a, %c5) : memref<?xi8, #5>
%thread_a_odd = miopen.alloc(%blockwise_copy_matrix_a, %c5) : memref<?xi8, #5>
%thread_b_even = miopen.alloc(%blockwise_copy_matrix_b, %c5) : memref<?xi8, #5>
%thread_b_odd = miopen.alloc(%blockwise_copy_matrix_b, %c5) : memref<?xi8, #5>

# zero-init %thread_c
miopen.fill(%thread_c, %c0) : memref<?xf32, #5>

# copy from global (generic tensor) to LDS (naive tensor).
miopen.blockwise_copy(%matrix_a, %block_a_even) : memref<?x?xf32>, memref<?xi8, #3> 
miopen.blockwise_copy(%matrix_b, %block_b_even) : memref<?x?xf32>, memref<?xi8, #3>

# %total_iteration = k / (k_per_block * 2)
loop.for %iter = %c0 to %total_iteration {

  # manually unrolled double buffered loop.

  miopen.lds_barrier()

  # copy from global (generic tensor) to register (naive tensor).
  miopen.blockwise_copy(%matrix_a, %thread_a_even) { move_source_offset = k_per_block } : memref<?x?xf32>, memref<?xi8, #5>
  miopen.blockwise_copy(%matrix_b, %thread_b_even) { move_source_offset = k_per_block } : memref<?x?xf32>, memref<?xi8, #5>

  # blockwise GEMM is currently always LDS * LDS to register.
  miopen.blockwise_gemm(%matrix_block_a_even, %matrix_block_b_even, %matrix_thread_c) {
    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?xf32, #3>, memref<?x?xf32, #3>, memref<?x?xf32, #5>

  # copy from register (naive tensor) to LDS (naive tensor).
  miopen.blockwise_copy(%thread_a_even, %block_a_odd) : memref<?xi8, #5>, memref<?xi8, #3>
  miopen.blockwise_copy(%thread_b_even, %block_b_odd) : memref<?xi8, #5>, memref<?xi8, #3>


  miopen.lds_barrier()

  # copy from global (generic tensor) to register (naive tensor).
  miopen.blockwise_copy(%matrix_a, %thread_a_odd) { move_source_offset = k_per_block } : memref<?x?xf32>, memref<?xi8, #5>
  miopen.blockwise_copy(%matrix_b, %thread_b_odd) { move_source_offset = k_per_block } : memref<?x?xf32>, memref<?xi8, #5>

  # blockwise GEMM is currently always LDS * LDS to register.
  # matrix A, B, C are all naive tensors.
  miopen.blockwise_gemm(%matrix_block_a_odd, %matrix_block_b_odd, %matrix_thread_c) {
    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?xf32, #3>, memref<?x?xf32, #3>, memref<?x?xf32, #5>

  # copy from register (naive tensor) to LDS (naive tensor).
  miopen.blockwise_copy(%thread_a_even, %block_a_even) : memref<?xi8, #5>, memref<?xi8, #3>
  miopen.blockwise_copy(%thread_b_even, %block_b_even) : memref<?xi8, #5>, memref<?xi8, #3>
}

# loop tail
%has_one_iteration_left = (k % (k_per_block * 2) != 0
miopen.lds_barrier()
loop.if %has_one_iteration_left {
  miopen.blockwise_gemm(%matrix_block_a_even, %matrix_block_b_even, %matrix_thread_c) {
    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?xf32, #3>, memref<?x?xf32, #3>, memref<?x?xf32, #5>
} else {
  miopen.blockwise_gemm(%matrix_block_a_odd, %matrix_block_b_odd, %matrix_thread_c) {
    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?xf32, #3>, memref<?x?xf32, #3>, memref<?x?xf32, #5>
}

# copy from register (naive tensor) to global (generic tensor)
miopen.threadwise_copy(%thread_c, %matrix_c) : memref<?xi8, #5>, memref<?x?xf32>
```

-------------------------------------------------------------------------------

Blockwise GEMM -> Threadwise Slice Copy + Threadwise GEMM

```mlir
miopen.blockwise_gemm(%block_a, %block_b, %thread_c) {
  m_per_thread = 64,
  n_per_thread = 64,
  k_per_thread = 16,

  m_level0_cluster = 16,
  n_level0_cluster = 16,
  m_level1_cluster = 16,
  n_level1_cluster = 16,

  matrix_a_source_data_per_read = 4,
  matrix_b_source_data_per_read = 4
}
```


```mlir
# naive version
# non-XDLOPS

# %threadwise_matrix_a is computed from k_per_thread_loop and m_per_thread
%thread_a = miopen.alloc(%threadwise_matrix_a, %c5) : memref<?xi8, #5>
%thread_b = miopen.alloc(%threadwise_matrix_b, %c5) : memref<?xi8, #5>

%total_iteration = %K / %k_per_thread_loop

# unroll
loop.for %iter_k = 0 to %total_iteration {
  # read Matrix A
  # unroll
  loop.for %iter_a = 0 to %m_per_thread / (%m_per_thread_sub_c * %m_level0_thread_cluster * %m_level1_thread_cluster) {
    # copy from LDS (naive tensor) to register (naive tensor)
    miopen.threadwise_copy(%block_a, %thread_a) { offset_block = TBD, offset_thread = TBD }
  }

  # read Matrix B
  # unroll
  loop.for %iter_b = 0 to %n_per_thread / (%n_per_thread_sub_c * %n_level0_thread_cluster * %n_level1_thread_cluster) {
    # copy from LDS (naive tensor) to register (naive tensor)
    miopen.threadwise_copy(%block_b, %thread_b) { offset_block = TBD, offset_thread = TBD }
  }

  # C += A * B
  # A, B, C are all on registers (naive tensor)
  miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c)
}
```

```mlir
# pipelined 2x2 version
# non-XDLOPS

# %threadwise_matrix_a is computed from k_per_thread_loop and m_per_thread
%thread_a = miopen.alloc(%threadwise_matrix_a, %c5) : memref<?xi8, #5>
# %threadwise_matrix_b is computed from k_per_thread_loop and n_per_thread
%thread_b = miopen.alloc(%threadwise_matrix_b, %c5) : memref<?xi8, #5>

# read A_sub_0
# copy from LDS (naive tensor) to register (naive tensor)
miopen.threadwise_copy(%block_a, %thread_a) { offset_source = TBD }

# read B_sub_0
# copy from LDS (naive tensor) to register (naive tensor)
miopen.threadwise_copy(%block_b, %thread_b) { offset_source = TBD }

# read B_sub_1
# copy from LDS (naive tensor) to register (naive tensor)
miopen.threadwise_copy(%block_b, %thread_b) { offset_source = TBD, offset_dest = TBD }

# read A_sub_1
# copy from LDS (naive tensor) to register (naive tensor)
miopen.threadwise_copy(%block_a, %thread_a) { offset_source = TBD, offset_dest = TBD }

# C_sub_00 += transpose(A_sub_0) * B_sub_0
# A, B, C are all on registers (naive tensor)
miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c)

# C_sub_01 += transpose(A_sub_0) * B_sub_1
# A, B, C are all on registers (naive tensor)
miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c) { offset_b = TBD, offset_c = TBD }


%total_iteration = %K / %k_per_thread_loop
loop.for %iter_k = 0 to %total_iteration {
  # read A_sub_0
  # copy from LDS (naive tensor) to register (naive tensor)
  miopen.threadwise_copy(%block_a, %thread_a) { offset_source = TBD }

  # C_sub_10 += transpose(A_sub_1) * B_sub_0
  # A, B, C are all on registers (naive tensor)
  miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c) { offset_a = TBD, offset_c = TBD }

  # read B_sub_0
  # copy from LDS (naive tensor) to register (naive tensor)
  miopen.threadwise_copy(%block_b, %thread_b) { offset_source = TBD }

  # C_sub_11 += transpose(A_sub_1) * B_sub_1
  # A, B, C are all on registers (naive tensor)
  miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c) { offset_a = TBD, offset_b = TBD, offset_c = TBD }

  # read B_sub_1
  # copy from LDS (naive tensor) to register (naive tensor)
  miopen.threadwise_copy(%block_b, %thread_b) { offset_source = TBD, offset_dest = TBD }

  # read A_sub_1
  # copy from LDS (naive tensor) to register (naive tensor)
  miopen.threadwise_copy(%block_a, %thread_a) { offset_source = TBD, offset_dest = TBD }

  # C_sub_00 += transpose(A_sub_0) * B_sub_0
  # A, B, C are all on registers (naive tensor)
  miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c)

  # C_sub_01 += transpose(A_sub_0) * B_sub_1
  # A, B, C are all on registers (naive tensor)
  miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c) { offset_b = TBD, offset_c = TBD }
}


# C_sub_10 += transpose(A_sub_1) * B_sub_0
# A, B, C are all on registers (naive tensor)
miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c) { offset_a = TBD, offset_c = TBD }

# C_sub_11 += transpose(A_sub_1) * B_sub_1
# A, B, C are all on registers (naive tensor)
miopen.threadwise_gemm(%thread_a, %thread_b, %thread_c) { offset_a = TBD, offset_b = TBD, offset_c = TBD }

}

```

Optimization
============

```
// Merge + PassThroguh -> Unfold

def : Pat<(TransformOp: $op, $A, $B),
          (UnfoldOp),
          [
            (Constraint<IsOpOfType($A, "MergeOp")),
            (Constraint<IsOpOfType($B, "PassThrough")),
          ],
          (addBenefit 1)>;
```

-------------------------------------------------------------------------------

Output
======
- C++ logic for solver (.cpp)
- C++ logic for kernel wrapper (.cpp)
- C++ logic for kernel algorithm (.hpp)

-------------------------------------------------------------------------------

Compilation + Execution
=======================
- Output source codes are fed into MIOpen build directory under /tmp.
- Invoke HipBuild() via new hidden MIOpen API.

Tunable paremeters and their rules for non-XDLOPs kernels
=========================================================

truly need to be tuned:

CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK: 128
CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK: 128
CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK: 16

CK_PARAM_TUNABLE_GEMM_M_PER_THREAD_SUB_C: 4
CK_PARAM_TUNABLE_GEMM_N_PER_THREAD_SUB_C: 4


assume: 2x2 pipeline


derivable:

CK_PARAM_TUNABLE_GEMM_M_LEVEL0_CLUSTER: 4
CK_PARAM_TUNABLE_GEMM_M_LEVEL1_CLUSTER: 4
CK_PARAM_TUNABLE_GEMM_N_LEVEL0_CLUSTER: 4
CK_PARAM_TUNABLE_GEMM_N_LEVEL1_CLUSTER: 4
  - derived from *_PER_BLOCK and *PER_THREAD_SUB_C
  - constraint: M_PER_THREAD_SUB_C * M_LEVEL0_CLUSTER * M_LEVEL1_CLUSTER * 2(pipeline depth) = M_PER_BLOCK. same for N.

CK_PARAM_TUNABLE_BLOCK_SIZE: 256
  - M_LEVEL0_CLUSTER * M_LEVEL1_CLUSTER * M_LEVEL0_CLUSTER * N_LEVEL1_CLUSTER

CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K: 2
CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M: 128
  - constraint: COPY_CLUSTER_LENGTHS_GEMM_K * GEMM_M = BLOCK_SIZE

CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M: 1
CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K: 1
  - vary per layout, TBD

CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K: 2
CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N: 128
  - constraint: COPY_CLUSTER_LENGTHS_GEMM_K * GEMM_M = BLOCK_SIZE

CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N: 1
CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N: 1
  - vary per layout, TBD

CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1: 1
  - vary per layout, TBD

Tunable parameters and their rules for XDLOPs kernels
=====================================================
truly need to be tuned:
CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK: 128
CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK: 128
CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK: 16

CK_PARAM_GEMM_M_PER_WAVE: 64
CK_PARAM_GEMM_N_PER_WAVE: 64


derivable:

CK_PARAM_TUNABLE_BLOCK_SIZE: 256
- M_PER_BLOCK / M_PER_WAVE = # of wavefronts on M dimension
- N_PER_BLOCK / N_PER_WAVE = # of wavefronts on N dimension
- # of wavefronts on M dimension *
  # of wavefronts on N dimension * wavefront size (64) = BLOCK_SIZE

CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K: 2
CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M: 128
CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M: 1
CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K: 1
- rule same as non-XDLOP version

CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K: 2 
CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N: 128
CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N: 1
CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N: 1
- rule same as non-XDLOP version

CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DATA_PER_ACCESS_N: 1
- fixed at 1 for now
