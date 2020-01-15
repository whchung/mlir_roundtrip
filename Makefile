LLVM_ROOT=~/llvm-project/build/bin
CLANG=$(LLVM_ROOT)/clang++
MLIR_OPT=$(LLVM_ROOT)/mlir-opt
MLIR_TRANSLATE=$(LLVM_ROOT)/mlir-translate
LLVM_LINK=$(LLVM_ROOT)/llvm-link
LLVM_OPT=$(LLVM_ROOT)/opt
HIPCC=/opt/rocm/hip/bin/hipcc

all: matmul matmul_tiled vecadd vecadd_gpu matmul_gpu conv conv_gpu conv_gpu_gridwise_gemm

matmul: matmul_lib.cpp matmul.mlir main.cpp
	$(CLANG) -O3 -emit-llvm -S matmul_lib.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o matmul_lib.ll
	$(MLIR_OPT) matmul.mlir -convert-linalg-to-llvm -convert-vector-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > matmul.ll
	$(LLVM_LINK) -S -o linked.ll matmul_lib.ll matmul.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DMATMUL main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o matmul

matmul_tiled: matmul_lib.cpp matmul.mlir main.cpp
	$(CLANG) -O3 -emit-llvm -S matmul_lib.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o matmul_lib.ll
	$(MLIR_OPT) matmul.mlir -test-linalg-transform-patterns -convert-linalg-to-llvm -convert-vector-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > matmul.ll
	$(LLVM_LINK) -S -o linked.ll matmul_lib.ll matmul.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DMATMUL main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o matmul_tiled

clean:
	rm -f *.ll matmul matmul_tiled vecadd vecadd_gpu matmul_gpu conv conv_gpu conv_gpu_gridwise_gemm *.so

vecadd: vecadd_lib.cpp vecadd.mlir main.cpp
	$(CLANG) -O3 -emit-llvm -S vecadd_lib.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o vecadd_lib.ll
	$(MLIR_OPT) vecadd.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > vecadd.ll
	$(LLVM_LINK) -S -o linked.ll vecadd_lib.ll vecadd.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DVECADD main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o vecadd

librocm_wrappers.so: rocm_wrappers.cpp
	$(HIPCC) -shared -fPIC -I /opt/rocm/rocblas/include -I /opt/rocm/miopen/include rocm_wrappers.cpp -lrocblas -lMIOpen -o librocm_wrappers.so

vecadd_gpu: librocm_wrappers.so
	$(CLANG) -O3 -emit-llvm -S rocm_bridge.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o rocm_bridge.ll
	$(MLIR_OPT) vecadd_gpu.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > vecadd_gpu.ll
	$(LLVM_LINK) -S -o linked.ll rocm_bridge.ll vecadd_gpu.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DVECADD main.cpp opt.ll librocm_wrappers.so ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -Wl,-rpath=. -o vecadd_gpu

matmul_gpu: librocm_wrappers.so rocm_bridge.cpp matmul_gpu.mlir main.cpp
	$(CLANG) -O3 -emit-llvm -S rocm_bridge.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o rocm_bridge.ll
	$(MLIR_OPT) matmul_gpu.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > matmul_gpu.ll
	$(LLVM_LINK) -S -o linked.ll rocm_bridge.ll matmul_gpu.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DMATMUL main.cpp opt.ll librocm_wrappers.so ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -Wl,-rpath=. -o matmul_gpu

conv: conv_lib.cpp conv.mlir main.cpp
	$(CLANG) -O3 -emit-llvm -S conv_lib.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o conv_lib.ll
	$(MLIR_OPT) conv.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > conv.ll
	$(LLVM_LINK) -S -o linked.ll conv_lib.ll conv.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DCONV main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o conv

conv_gpu: librocm_wrappers.so rocm_bridge.cpp conv_gpu.mlir main.cpp
	$(CLANG) -O3 -emit-llvm -S rocm_bridge.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o rocm_bridge.ll
	$(MLIR_OPT) conv_gpu.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > conv_gpu.ll
	$(LLVM_LINK) -S -o linked.ll rocm_bridge.ll conv_gpu.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DCONV main.cpp opt.ll librocm_wrappers.so ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -Wl,-rpath=. -o conv_gpu

# env vars to force MIOpen use gridwise GEMM:
# export MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW=0
# export MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES=0
# export MIOPEN_DEBUG_GCN_ASM_KERNELS=0
# export MIOPEN_DEBUG_CONV_FFT=0
# export MIOPEN_DEBUG_CONV_DIRECT=0
# export MIOPEN_DEBUG_CONV_GEMM=0
# export MIOPEN_DEBUG_CONV_SCGEMM=0
# export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
#
# two scripts have been created to help set the env vars:
#
# set env vars:
# source force_gridwise_gemm.sh
#
# unset env vars:
# source unset_gridwise_gemm.sh
conv_gpu_gridwise_gemm: librocm_wrappers.so rocm_bridge.cpp conv_gpu_gridwise_gemm.mlir main.cpp
	$(CLANG) -O3 -emit-llvm -S rocm_bridge.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o rocm_bridge.ll
	$(MLIR_OPT) conv_gpu_gridwise_gemm.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > conv_gpu_gridwise_gemm.ll
	$(LLVM_LINK) -S -o linked.ll rocm_bridge.ll conv_gpu_gridwise_gemm.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DCONV main.cpp opt.ll librocm_wrappers.so ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -Wl,-rpath=. -o conv_gpu_gridwise_gemm
