#include <cstdio>

__global__ void helloFromGPU(void){
  printf("Hello world from GPU\n");
}

int main(void){
  printf("Hello world from CPU\n");
  helloFromGPU<<<1,10>>>();
  cudaDeviceReset();
  return 0;
}



