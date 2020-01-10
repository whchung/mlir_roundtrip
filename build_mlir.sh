#!/bin/bash

if [ "$1" != "" ]; then
    echo "Launch tiled version."
    # tiled version.
    ~/llvm-project/build/bin/mlir-opt matmul.mlir \
      -test-linalg-transform-patterns \
      -convert-linalg-to-llvm \
      -convert-vector-to-llvm \
    | ~/llvm-project/build/bin/mlir-translate -mlir-to-llvmir > matmul.ll
else
    echo "Launch non-tiled version."
    # non-tiled version.
    ~/llvm-project/build/bin/mlir-opt matmul.mlir \
      -convert-linalg-to-llvm \
      -convert-vector-to-llvm \
    | ~/llvm-project/build/bin/mlir-translate -mlir-to-llvmir > matmul.ll
fi
