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

-------------------------------------------------------------------------------

Transformed ops
===============

An example based on NCHW/KCYX/NKHW:

```mlir
// filter tensor
%filter_gemmK_gemmM = miopen.transform(%filter) {
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
  ]
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
  ]
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
  ]
} : memref<?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32>

%input_gemmK_gemmN = miopen.transform(%input_n_c_y_ho_x_wo) {
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
  ]
} : memref<?x?x?x?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// output tensor
%output_gemmM_gemmN = miopen.transform(%output) {
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
  ]
} : memref<?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// apply gridwise GEMM
miopen.gridwise_gemm(%filter_gemmK_gemmM, %input_gemmK_gemmN, %output_gemmM_gemmN) {
  parameters = [
    // tuning parameters
  ]
} : memref<?x?xf32>,
    memref<?x?xf32>,
    memref<?x?xf32>
```

-------------------------------------------------------------------------------

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
