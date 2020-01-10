#!/bin/bash

if [ "$1" != "" ]; then
  ~/llvm-project/build/bin/clang++ -O3 main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o $1
else
  ~/llvm-project/build/bin/clang++ -O3 main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o output
fi
