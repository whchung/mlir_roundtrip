all:
	./build_blas.sh
	./build_mlir.sh
	./ir-link.sh
	./ir-opt.sh
	./output.sh

tiled:
	./build_blas.sh
	./build_mlir.sh t
	./ir-link.sh
	./ir-opt.sh
	./output.sh output_tiled

clean:
	rm -f *.ll output output_tiled

