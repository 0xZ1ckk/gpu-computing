#include <stdio.h>
#include <stdlib.h>
#define N 10


__global__ void add_vect(int *a, int *b, int *c){
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

int main(){
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int nBytes = N * sizeof(int);

    a = (int *) malloc(nBytes);
    for(int i = 0; i < N; i++){
        a[i] = rand();
        printf("%d\n", a[i]);
    }
    printf("-------------\n");
    b = (int *) malloc(nBytes);
    for(int i = 0; i < N; i++){
        b[i] = rand();
        printf("%d\n", b[i]);
    }
    c = (int *) malloc(nBytes);

    cudaMalloc((void**) &dev_a, nBytes);
    cudaMalloc((void**) &dev_b, nBytes);
    cudaMalloc((void**) &dev_c, nBytes);

    cudaMemcpy(dev_a, a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, nBytes, cudaMemcpyHostToDevice);

    add_vect<<<1, N>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, nBytes, cudaMemcpyDeviceToHost);
    printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");

    for(int i = 0; i < N; i++){
        printf("%d\n", c[i]);
    }

    free(a);
    free(b);
    free(c);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
