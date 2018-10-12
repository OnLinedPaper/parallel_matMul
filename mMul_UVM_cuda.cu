
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <stdio.h>

__global__ //set global so this runs on the GPU
void multiply(float *a, float *b, float *c, int aRow, int aCol, 
  int bCol, const float aSize, const float bSize) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    int block = blockIdx.x;


    for(int i = 0; i < aRow; i += 1)
      for(int j = index + (block * stride); j < bCol; j+= stride)
        for(int k = 0; k < aCol; k += 1)
          c[i*bCol + j] += a[i*aCol + k] * b[j + k*bCol];  

  __syncthreads(); //wait for all threads to finish
  return;
}

void scalar_multiply(float *a, float *b, float *c, int aRow, int aCol,
  int bCol) {
    //run this on cpu to check for errors

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

  int errorcheck = 0;
  int threads = 512;
  int blocks = 512;
  int DEBUG = 0;

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

  //malloc shared memory that can be accessed via GPU and CPU
  cudaMallocManaged(&cu_a, aSize);
  cudaMallocManaged(&cu_b, bSize);
  cudaMallocManaged(&cu_c, cSize);

  //initialize them to randoms
  for(int i = 0; i < aRow*aCol; i++) {
    a[i] = cu_a[i] = rand() % 1000;
  }
  for(int i = 0; i < bRow*bCol; i++) {
    b[i] = cu_b[i] = rand() % 1000;
  }
  for(int i = 0; i < aRow*bCol; i++) {
    c[i] = cu_c[i] = 0;
  }

  //warmup
  std::cout << "warming up...\n";
  multiply<<<blocks, threads>>>(cu_a, cu_b, cu_c, aRow, aCol, bCol, aSize, bSize);
  
  //after warming up, set memory back to 0
  cudaMemset(cu_c, 0, cSize);

  std::cout << "done.\nrunning tests...\n";

  if(errorcheck){
    //run a CPU version to check for errors
    scalar_multiply(a, b, c, aRow, aCol, bCol);
  }


  double fulltime = 0;
  int repeats = 1;
  for(int i=0; i<repeats; i++) {
   
    //reset memory to zeros
    cudaMemset(cu_c, 0, cSize);

    std::clock_t start = std::clock();
    
    multiply<<<blocks, threads>>>(cu_a, cu_b, cu_c, aRow, aCol, bCol, aSize, bSize);

    //wait for all threads to finish before "timing" the code
    cudaDeviceSynchronize();
    
    std::clock_t end = std::clock();
    
    fulltime += (end - start);
    
  }

  if(DEBUG) {
    //print every entry
    for(int i=0; i<aRow*bCol; i++)
      std::cerr << "c[" << i << "]\t" << (c[i] == cu_c[i] ? "\x1B[32mPASS\x1B[0m\t" : "\x1B[31mFAIL\x1B[0m\t") << c[i] << " " << cu_c[i] << std::endl;
  }

  int arraycheck = 1;
  if(errorcheck) {
    //run error checking
    for(int i=0; i<aRow*bCol; i++) 
      if(c[i] != cu_c[i])
        arraycheck = 0;

  std::cout << (arraycheck ? "\x1B[32mPASS\x1B[0m" : "\x1B[31mFAIL\x1B[0m") << std::endl;
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

  //free shared memory
  cudaFree(cu_a);
  cudaFree(cu_b);
  cudaFree(cu_c);

  //free cpu memory
  free(a);
  free(b);
  free(c);

  return 0;
}
