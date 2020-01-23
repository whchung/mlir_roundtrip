High-level op
=============

```mlir
linalg.conv -> miopen.conv

miopen.conv(%filter, %input, %output) {
    filter_layout = [0: k, 1: c, 2: y, 3: x],
    input_layout = [0: n, 1: c, 2: h, 3: w],
    output_layout = [0: n, 1: k, 2: ho, 3: wo],
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
    0: merge(1: c, 2: y, 3: x), gemmK
    1: passthrough(0: k), gemmM
  ]
} : memref<?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// input tensor
%input_n_c_hipad_wipad = miopen.transform(%input) {
  layout = [
    0: passthrough(0: n), n
    1: passthrough(1: c), c
    2: pad(2: h, 0, 0), hipad
    3: pad(3: w, 0, 0), wipad
  ]
} : memref<?x?x?x?xf32> to memref<?x?x?x?f32>

%input_n_c_y_ho_x_wo = miopen.transform(%input_n_c_hipad_wipad) {
  layout = [
    0: passthrough(0: n), n
    1: passthrough(1: c), c
    [2, 3]: embed(2: hipad, 2, [1, 1, 0]), [y, ho]
    [4, 5]: embed(3: wipad, 2, [1, 1, 0]), [x, wo]
  ]
} : memref<?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32>

%input_gemmK_gemmN = miopen.transform(%input_n_c_y_ho_x_wo) {
  layout = [
    0: merge(1: c, 2: y, 4: x), gemmK
    1: merge(0: n, 3: ho, 5: wo), gemmN
  ]
} : memref<?x?x?x?x?x?x?xf32> to memref<?x?xf32>
```

```mlir
// output tensor
%output_gemmM_gemmN = miopen.transform(%output) {
  layout = [
    0: passthrough(1: k), gemmM
    1: merge(0: n, 2: ho, 3: wo), gemmN
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

