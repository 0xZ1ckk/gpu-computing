#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
      exit(-10 * error);                                                       \
    }                                                                          \
  }

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float *ip, int size) {
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
  for (int idx = 0; idx < N; idx++)
    C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;
  for (int i = 0; i < N; i++) {
    if (fabsf(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("Arrays do not match!\n");
      printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }
  if (match)
    printf("Arrays match.\n\n");
}

int main(int argc, char **argv) {
  printf("%s Starting...\n", argv[0]);

  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  int nElem = 1 << 24;
  printf("Vector size %d\n", nElem);

  size_t nBytes = nElem * sizeof(float);
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  double iStart, iElaps;

  iStart = cpuSecond();
  initialData(h_A, nElem);
  initialData(h_B, nElem);
  iElaps = cpuSecond() - iStart;
  printf("initData Time elapsed %f sec\n", iElaps);

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  iStart = cpuSecond();
  sumArraysOnHost(h_A, h_B, hostRef, nElem);
  iElaps = cpuSecond() - iStart;
  printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

  float *d_A, *d_B, *d_C;
  CHECK(cudaMalloc((float **)&d_A, nBytes));
  CHECK(cudaMalloc((float **)&d_B, nBytes));
  CHECK(cudaMalloc((float **)&d_C, nBytes));

  CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

  int iLen = 1024;
  dim3 block(iLen);
  dim3 grid((nElem + block.x - 1) / block.x);

  iStart = cpuSecond();
  sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  printf("sumArraysOnGPU <<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
  checkResult(hostRef, gpuRef, nElem);

  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  return 0;
}

