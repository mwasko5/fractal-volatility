#include "kernel.cuh"

#define NUM_ELEMENTS 16384

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

__global__ void volatility_optimized(float initial_volatility, float* bins, float* random_seeds) {
    // bins = array of the volatility levels (normally distributed) (so initial volatility will be in the middle)
    // random_seeds = array of how much to multiply the previous data by (0-1 but most commonly 0.5)
    // NEED TO LOOK INTO cuRAND
    __shared__ float seed_device_shared[4096];
    for (int i = 0; i < 4096; i++) {
        seed_device_shared[i] = random_seeds[i];
    }
	
	__syncthreads();

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
		    summation = (summation * seed_device_shared[indexing_left % 4096]);
		    indexing_left -= 1;
        }

        bins[i] = summation * seed_device_shared[i % 4096];
    }
    else if (i > middle_index) {
        int indexing_right = middle_index + 1;
        float summation = initial_volatility;

        while (indexing_right < i) {
		    summation = (summation * seed_device_shared[indexing_right % 4096]);
			indexing_right += 1;
        }

        bins[i] = summation * seed_device_shared[i % 4096];
    }
}
