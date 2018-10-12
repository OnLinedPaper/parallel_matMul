#include <iostream>
#include <cstdlib>
#include <ctime>

#include <unistd.h>
#include <stdio.h>
#include <omp.h>

#define DEBUG 0

void scalar_multiply(float *a, float *b, float *c, int aRow, int aCol,
  int bCol) {
  
  int nthreads, tid;
  omp_set_dynamic(0);
  omp_set_num_threads(4096);
#pragma omp parallel shared (a, b, c) private(nthreads, tid) //shared memory a, b, c
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
#pragma omp for //parallelize one for loop - reduces runtime from O(n^3) to O(n^2)
    for(int i = 0; i < aRow; ++i)
      for(int j = tid; j < bCol; j+= nthreads)
        for(int k = 0; k < aCol; ++k)
          c[i*bCol + j] += a[i*aCol + k] * b[j + k*bCol];
  }
}


int main(int argc, char *argv[])
{


  if(argc < 5) {
    std::cerr << "usage: " << argv[0] << " aRows aCols bRows bCols\n";
    return(-1);
  }

  int errorcheck = 0; //set to 0 to skip error checking
  int threads = 256;

  if(atoi(argv[2]) != atoi(argv[3])) {
    std::cerr << "error! aCols must match bRows. " << 
      argv[2] << ", " << argv[3] << std::endl;
    return(-1);
  }

  srand(4); //so creative

  //accept 4 args: row, col, for a and b
  float aRow = atoi(argv[1]);
  float aCol = atoi(argv[2]);
  float bRow = atoi(argv[3]);
  float bCol = atoi(argv[4]);
  float cRow = aRow;
  float cCol = bCol;

  float *a = (float *)malloc(aRow * aCol * sizeof(float));
  float *b = (float *)malloc(bRow * bCol * sizeof(float));
  float *c = (float *)malloc(cRow * cCol * sizeof(float));



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

  //warmup
  std::cout << "warming up...\n";

  std::cout << "done.\nrunning tests...\n";

  if(errorcheck){
    scalar_multiply(a, b, c, aRow, aCol, bCol);
  }

  std::clock_t start = std::clock();

  float repeats = 4;
  for(int i=0; i<repeats; i++) {
    //repeat code in case matrix sizes are too small to time
    scalar_multiply(a, b, c, aRow, aCol, bCol);
  }

  std::clock_t end = std::clock();

//error checking not implemented for OpenMP
//  int arraycheck = 1;
//  if(errorcheck) {
    
//    std::cout << (arraycheck ? "\x1B[32mPASS\x1B[0m" : "\x1B[31mFAIL\x1B[0m") << std::endl;
//  }
    
  float flops = (aRow*aCol*bCol*2);
  double s_time = ((end - start) / (double)(CLOCKS_PER_SEC));

  std::cout << "a[" << aRow << "," << aCol << "], b[" << bRow << "," << bCol << "], c[" << cRow << "," << cCol << "]\n";
  std::cout << "time: " << s_time*1000 << "ms\n";
  std::cout << "performance: " << flops << " flops at " << ((flops / 1000000000) / (s_time / repeats)) << "GFlop/s\n";

  if(DEBUG) {
  //printout
    for(int i=0; i<aRow * bCol; i++)
      std::cout << c[i] << " ";
    std::cout << std::endl; 
  }

  //free memory
  free(a);
  free(b);
  free(c);

  return 0;
}
