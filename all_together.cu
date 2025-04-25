#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

//#include "kernel.cuh"

#define NUM_ELEMENTS 4096
#define NUM_RANDOM 1024

#define SEED_MAX 0.99
#define SEED_MIN 0.95

#define BLOCK_SIZE 1024
#define GRID_SIZE NUM_ELEMENTS/BLOCK_SIZE

__constant__ float seed_device_constant[1024];

/*
    TODO: RESEARCH RANDOM NUMBER GENERATION IN GPU
        - could possible pre-process random numbers in CPU before sending to GPU in a shared memory array (optimzied version)
*/

__global__ void volatility_naive(float initial_volatility, float* bins, float* random_seeds) {
    // bins = array of the volatility levels (normally distributed) (so initial volatility will be in the middle)
    // random_seeds = array of how much to multiply the previous data by (0-1 but most commonly 0.5)
    // NEED TO LOOK INTO cuRAND

    // do all of this in global memory
    int i = threadIdx.x + blockIdx.x * blockDim.x; // global index

    const int middle_index = NUM_ELEMENTS / 2; // save as variable to save computational power
    
    if (i == middle_index) { 
        bins[middle_index] = initial_volatility;
    }
    
    __syncthreads();

    // figure out how to use thread index to do this position calculation
    if (i >= NUM_ELEMENTS) { 
        return;
    }

    if (i < middle_index) {
        int indexing_left = middle_index - 1;
        float summation = initial_volatility;

        while (indexing_left > i) {
            summation = (summation * random_seeds[indexing_left]);
            indexing_left -= 1;
        }

        bins[i] = summation * random_seeds[i];
    }
    else if (i > middle_index) {
        int indexing_right = middle_index + 1;
        float summation = initial_volatility;

        while (indexing_right < i) {
            summation = (summation * random_seeds[indexing_right]);
            indexing_right += 1;
        }

        bins[i] = summation * random_seeds[i];
    }
}

__global__ void volatility_optimized(float initial_volatility, float* bins) {
    // load random seeds into shared memory and bins
    // max constant memory is 1024 float values, just load the max amount and then do data manipulation on it

    //__shared__ float shared_bins[4096]; // this is the max size, need to do batches for optimized version if wanna use shared
    
    // algorithmic optimization idea: if the previous bin is 0, no sense in continuing computations and can end the loop
    int i = threadIdx.x + blockIdx.x * blockDim.x; // global index

    const int middle_index = NUM_ELEMENTS / 2; // save as variable to save computational power
    
    if (i == middle_index) { 
        bins[middle_index] = initial_volatility;
    }

    __syncthreads();

    // figure out how to use thread index to do this position calculation
    if (i >= NUM_ELEMENTS) { 
        return;
    }

    if (i < middle_index) {
        int indexing_left = middle_index - 1;
        float summation = initial_volatility;

        while (indexing_left > i) {
            if (summation < 0.0001) {
                break; // end doing trivial computations & conditional checks and just write the summation to all remaining bins
            }
            else {
                summation = (summation * seed_device_constant[indexing_left % 1024]);
                indexing_left -= 1;
            }
        }

        bins[i] = summation * seed_device_constant[i % 1024];
    }
    else if (i > middle_index) {
        int indexing_right = middle_index + 1;
        float summation = initial_volatility;

        while (indexing_right < i) {
            if (summation < 0.0001) {
                break; // end doing trivial computations & conditional checks and just write the summation to all remaining bins
            }
            else {
                summation = (summation * seed_device_constant[indexing_right % 1024]);
                indexing_right += 1;
            }
        }

        bins[i] = summation * seed_device_constant[i % 1024];
    }
    
}

void random_generator(float* random_bins, float min, float max) {
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < NUM_RANDOM; i++) {
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
    seed_host = (float*)malloc(NUM_RANDOM * sizeof(float));
	//seed_host = (float*)malloc(NUM_ELEMENTS * sizeof(float));
    random_generator(seed_host, SEED_MIN, SEED_MAX);
	
	/*
	for (int i = 0; i < 1024; i++) {
		printf("%d: %f\n", i, seed_host[i]);
	}
	*/  
	
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
    if (cudaMemcpyToSymbol(seed_device_constant, seed_host, 1024 * sizeof(float)) != cudaSuccess) { // seed device needs to be 1024 size
		printf("memcpy to constant memory error\n");
	} 

    cudaEventRecord(astartEvent, 0);
    
    //volatility_naive<<<blockDim, gridDim>>>(50, bins_device, seed_device);

    volatility_optimized<<<blockDim, gridDim>>>(1000000, bins_device);

    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    
    if (cudaMemcpy(bins_host, bins_device, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("bins memcpy error from device to host\n");
    }

    print_fractal(bins_host);
	  
	printf("Elapsed kernel execution time: %f ms\n", aelapsedTime);

    // free GPU and host memory
    cudaFree(bins_device);
    cudaFree(seed_device);

    free(bins_host);
    free(seed_host);

    return 0;
}