#include <stdio.h>
#define N 1000


__global__ void add_vect(int *a, int *b, int *c){
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

int main(){
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int nBytes = N * sizeof(int);

    a = (int *) malloc(nBytes);
    b = (int *) malloc(nBytes);
    c = (int *) malloc(nBytes);

    cudaMalloc((void**) &dev_a, nBytes);
    cudaMalloc((void**) &dev_b, nBytes);
    cudaMalloc((void**) &dev_c, nBytes);

    cudaMemcpy(dev_a, a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, nBytes, cudaMemcpyHostToDevice);

    add_vect<<<1, N>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, nBytes, cudaMemcpyDeviceToHost);

    printf("%d\n", c[3]);

    free(a);
    free(b);
    free(c);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
