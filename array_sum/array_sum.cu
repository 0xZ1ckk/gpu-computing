#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void sumArraysOnHost(float *arr1, float *arr2, float *arr3, int size) {
  for (int i = 0; i < size; i++) {
    arr1[i] = arr2[i] + arr3[i];
  }
}

void initalData(float *ip, int size) {
  time_t t;
  srand((unsigned int)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

int main(int argc, char **argv) {
  int nElem = 1024;
  size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *h_C;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);

  initalData(h_A, nElem);
  initalData(h_B, nElem);

  float *d_A, *d_B, *d_C;
  cudaMalloc((float **)&d_A, nBytes);
  cudaMalloc((float **)&d_B, nBytes);
  cudaMalloc((float **)&d_C, nBytes);
  printf("Address to array in device memory : %x\n", d_A);
  printf("Address to array in host memory : %x\n", h_A);

  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

  sumArraysOnHost<<<1,1>>>(d_A, d_B, d_C, nElem);

  cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

  cudaDeviceReset();

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  return (0);
}
