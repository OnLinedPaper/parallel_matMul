cudauvm: mMul_UVM_cuda.cu
	nvcc mMul_UVM_cuda.cu -o mMul

cudabasic: mMul_basic_cuda.cu
	nvcc mMul_basic_cuda.cu -o mMul

omp: mMul_openMP.cpp
	g++ -fopenmp mMul_openMP.cpp -o mMul

clean: 
	rm mMul
