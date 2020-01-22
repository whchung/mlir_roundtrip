High-level op
=============

linalg.conv -> miopen.conv

miopen.conv(%filter, %input, %output) {
    filter_layout = 'kcyx',
    input_layout = 'nchw',
    output_layout = 'nkhw',
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
} : memref<?x?x?x?xf32>,
    memref<?x?x?x?xf32>,
    memref<?x?x?x?xf32>

-------------------------------------------------------------------------------

Transformed ops
===============

#affine_map_filter = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] ->
                                (d1 * s1 + d2 * s2 + d3 * s3, d0)>
%filter_k_m = miopen.transform(%filter) #affine_map_filter : memref<?x?x?x?xf32> to memref<?x?xf32>


%input_k_n = miopen.transform(%input) {
    // TBD
} : memref<?x?x?x?xf32> to memref<?x?xf32>


#affine_map_output = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] ->
                                (d1, d0 * s0 + d2 * s2 + d3 * s3)>
%output_m_n = miopen.transform(%output)  #affine_map_output : memref<?x?x?x?xf32> to memref<?x?xf32>


miopen.gridwise_gemm(%filter_k_m, %input_k_n, %output_m_n) {
    // TBD
    // tuning parameters
} : memref<?x?xf32>,
    memref<?x?xf32>,
    memref<?x?xf32>

miopen.transform(%output_m_n) {
     // TBD
} : memref<?x?xf32> to memref<?x?x?x?xf32>

-------------------------------------------------------------------------------

Optimization
============

// Merge + PassThroguh -> Unfold

def : Pat<(TransformOp: $op, $A, $B),
          (UnfoldOp),
          [
            (Constraint<IsOpOfType($A, "MergeOp")),
            (Constraint<IsOpOfType($B, "PassThrough")),
          ],
          (addBenefit 1)>;

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

