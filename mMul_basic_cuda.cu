
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <stdio.h>

__global__ //declare global so the computer knows to run this on gpu
void multiply(float *a, float *b, float *c, int aRow, int aCol, 
  int bCol) {

    //nested for loop that "jumps" along blocks to perform multiplication
    int index = threadIdx.x;
    int stride = blockDim.x;
    int block = blockIdx.x;


    for(int i = 0; i < aRow; i += 1)
      for(int j = index + (block * stride); j < bCol; j+= stride)
        for(int k = 0; k < aCol; k += 1)
          c[i*bCol + j] += a[i*aCol + k] * b[j + k*bCol];  
  __syncthreads();
  return;
}

void scalar_multiply(float *a, float *b, float *c, int aRow, int aCol,
  int bCol) {
    //run without GPU work to use as error-checking
    for(int i = 0; i < aRow; ++i)
      for(int j = 0; j < bCol; ++j)
        for(int k = 0; k < aCol; ++k)
          c[i*bCol + j] += a[i*aCol + k] * b[j + k*bCol];
          return;
}


int main(int argc, char *argv[])
{

  if(argc < 5) {
    std::cerr << "usage: " << argv[0] << " aRows aCols bRows bCols\n";
    return(-1);
  }

  if(atoi(argv[2]) != atoi(argv[3])) {
    std::cerr << "error! aCols must match bRows. " << 
      argv[2] << ", " << argv[3] << std::endl;
    return(-1);
  }

  srand(4); //so creative

  int errorcheck = 0; //set to 1 to perform error checking
  int threads = 512;
  int blocks = 512;
  int DEBUG = 0; //set to 1 to print debug messages

  //accept 4 args: row, col, for a and b
  float aRow = atoi(argv[1]);
  float aCol = atoi(argv[2]);
  float bRow = atoi(argv[3]);
  float bCol = atoi(argv[4]);
  float cRow = aRow;
  float cCol = bCol;

  float aSize = aRow * aCol * sizeof(float);
  float bSize = bRow * bCol * sizeof(float);
  float cSize = cRow * cCol * sizeof(float);

  float *a = (float *)malloc(aSize);
  float *b = (float *)malloc(bSize);
  float *c = (float *)malloc(cSize);

  float *cu_a; 
  float *cu_b; 
  float *cu_c;

  //allocate memory on the GPU
  cudaMalloc(&cu_a, aSize);
  cudaMalloc(&cu_b, bSize);
  cudaMalloc(&cu_c, cSize);

  //initialize them to randoms
  for(int i = 0; i < aRow*aCol; i++) {
    a[i] = rand() % 1000;
  }
  for(int i = 0; i < bRow*bCol; i++) {
    b[i] = rand() % 1000;
  }
  for(int i = 0; i < aRow*bCol; i++) {
    c[i] = 0;
  }
    
  //put the data into the GPU
  cudaMemcpy(cu_a, a, aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(cu_b, b, bSize, cudaMemcpyHostToDevice);

  //warmup
  std::cout << "warming up...\n";
  multiply<<<blocks, threads>>>(cu_a, cu_b, cu_c, aRow, aCol, bCol);

  std::cout << "done.\nrunning tests...\n";

  if(errorcheck){
    //fill c with an array computed on cpu
    scalar_multiply(a, b, c, aRow, aCol, bCol);
  }


  double fulltime = 0;
  int repeats = 4;
  for(int i=0; i<repeats; i++) {
    //repeat in case the matrix size is too small to time properly

    std::clock_t start = std::clock();
    
    multiply<<<blocks, threads>>>(cu_a, cu_b, cu_c, aRow, aCol, bCol);
    cudaDeviceSynchronize();
   
    std::clock_t end = std::clock();
    cudaMemcpy(c, cu_c, cSize, cudaMemcpyDeviceToHost);
    
    fulltime += (end - start);
    
  }

    
  float flops = (aRow*aCol*bCol*2);
  double s_time = ((fulltime) / (double)(CLOCKS_PER_SEC));


  std::cout << "a[" << aRow << "," << aCol << "], b[" << bRow << "," << bCol << "], c[" << cRow << "," << cCol << "]\n";
  std::cout << "time: " << s_time*1000 << "ms\n";
  std::cout << "performance: " << flops << " flops at " << (((float)flops / 1000000000) / ((s_time) / repeats)) << "GFlop/s\n";

  if(DEBUG) {
  //printout
    for(int i=0; i<aRow * bCol; i++)
      std::cerr << c[i] << " ";
    std::cerr << std::endl; 
  }

  //free GPU memory
  cudaFree(cu_a);
  cudaFree(cu_b);
  cudaFree(cu_c);

  //free CPU memory
  free(a);
  free(b);
  free(c);

  return 0;
}
