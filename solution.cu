#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#include "kernel.cuh"

#define NUM_ELEMENTS 4096

#define SEED_MAX 0.9
#define SEED_MIN 0.7

#define BLOCK_SIZE 1024
#define GRID_SIZE NUM_ELEMENTS/BLOCK_SIZE

extern __constant__ float seed_device_constant[1024];

void random_generator(float* random_bins, float min, float max) {
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        random_bins[i] = min + ((float)rand() / RAND_MAX) * (max - min);
    }
}

void print_fractal(float* bins) {
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        printf("%d: %.4f\n ", i, bins[i]);
    }
    printf("\n");
}

int main(void) {
    // timing related things
    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);

    // initialize arrays
    float* seed_host;
    float* seed_device;

    float* bins_host;
    float* bins_device;

    // generate random seeds
    seed_host = (float*)malloc(NUM_ELEMENTS * sizeof(float));
    random_generator(seed_host, SEED_MIN, SEED_MAX);

    // allocate memory and initialize host_bins
    bins_host = (float*)malloc(NUM_ELEMENTS * sizeof(float));
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		bins_host[i] = 0;
	}
    
    // memcpy the seeds to GPU and malloc the bins on device
    if (cudaMalloc((void **)&bins_device, NUM_ELEMENTS * sizeof(float)) != cudaSuccess) {
        printf("bins malloc error\n");
    }

    if (cudaMalloc((void **)&seed_device, NUM_ELEMENTS * sizeof(float)) != cudaSuccess) {
        printf("seed malloc error\n");
    }

    if (cudaMemcpy(seed_device, seed_host, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("seed memcpy error from host to device\n");
    }
	
	if (cudaMemcpy(bins_device, bins_host, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("bins memcpy error from host to device\n");
    }

    dim3 blockDim(BLOCK_SIZE), gridDim(GRID_SIZE);

    // copy constant memory to GPU for optimized
    cudaMemcpyToSymbol(seed_device_constant, &seed_host, 1024 * sizeof(float)); // seed device needs to be 1024 size

    cudaEventRecord(astartEvent, 0);
    
    //volatility_naive<<<blockDim, gridDim>>>(50, bins_device, seed_device);

    volatility_optimized<<<blockDim, gridDim>>>(50, bins_device);

    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    
    if (cudaMemcpy(bins_host, bins_device, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("bins memcpy error from device to host\n");
    }

    //print_fractal(bins_host);
    printf("Elapsed kernel execution time: %f", aelapsedTime);

    // free GPU and host memory
    cudaFree(bins_device);
    cudaFree(seed_device);

    free(bins_host);
    free(seed_host);

    return 0;
}