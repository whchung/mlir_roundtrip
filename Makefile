LLVM_ROOT=~/llvm-project/build/bin
CLANG=$(LLVM_ROOT)/clang++
MLIR_OPT=$(LLVM_ROOT)/mlir-opt
MLIR_TRANSLATE=$(LLVM_ROOT)/mlir-translate
LLVM_LINK=$(LLVM_ROOT)/llvm-link
LLVM_OPT=$(LLVM_ROOT)/opt
HIPCC=/opt/rocm/hip/bin/hipcc

all:
	$(CLANG) -O3 -emit-llvm -S blas.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o blas.ll
	$(MLIR_OPT) matmul.mlir -convert-linalg-to-llvm -convert-vector-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > matmul.ll
	$(LLVM_LINK) -S -o linked.ll blas.ll matmul.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DMATMUL main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o matmul

tiled:
	$(CLANG) -O3 -emit-llvm -S blas.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o blas.ll
	$(MLIR_OPT) matmul.mlir -test-linalg-transform-patterns -convert-linalg-to-llvm -convert-vector-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > matmul.ll
	$(LLVM_LINK) -S -o linked.ll blas.ll matmul.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DMATMUL main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o matmul_tiled

clean:
	rm -f *.ll matmul matmul_tiled vecadd vecadd_gpu matmul_gpu *.so

vecadd:
	$(CLANG) -O3 -emit-llvm -S external_lib.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o external_lib.ll
	$(MLIR_OPT) vecadd.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > vecadd.ll
	$(LLVM_LINK) -S -o linked.ll external_lib.ll vecadd.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DVECADD main.cpp opt.ll ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -o vecadd

libhip_wrappers.so: hip_wrappers.cpp
	$(HIPCC) -shared -fPIC -I /opt/rocm/rocblas/include hip_wrappers.cpp -lrocblas -o libhip_wrappers.so

vecadd_gpu: libhip_wrappers.so
	$(CLANG) -O3 -emit-llvm -S hip_bridge.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o hip_bridge.ll
	$(MLIR_OPT) vecadd_gpu.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > vecadd_gpu.ll
	$(LLVM_LINK) -S -o linked.ll hip_bridge.ll vecadd_gpu.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DVECADD main.cpp opt.ll libhip_wrappers.so ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -Wl,-rpath=. -o vecadd_gpu

matmul_gpu: libhip_wrappers.so hip_bridge.cpp matmul_gpu.mlir main.cpp
	$(CLANG) -O3 -emit-llvm -S hip_bridge.cpp -std=c++14 -I ~/llvm-project/mlir/test/mlir-cpu-runner/include -o hip_bridge.ll
	$(MLIR_OPT) matmul_gpu.mlir -convert-linalg-to-llvm | $(MLIR_TRANSLATE) -mlir-to-llvmir > matmul_gpu.ll
	$(LLVM_LINK) -S -o linked.ll hip_bridge.ll matmul_gpu.ll
	$(LLVM_OPT) -S -O3 linked.ll -o opt.ll
	$(CLANG) -O3 -DMATMUL main.cpp opt.ll libhip_wrappers.so ~/llvm-project/mlir/test/mlir-cpu-runner/mlir_runner_utils.cpp -I ~/llvm-project/mlir/test/mlir-cpu-runner -Wl,-rpath=. -o matmul_gpu
