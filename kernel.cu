#include "kernel.cuh"

#define NUM_ELEMENTS 4096

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
    
    bins[middle_index] = initial_volatility;

    // figure out how to use thread index to do this position calculation
    if (i >= NUM_ELEMENTS) { 
        return;
    }

    if (i < middle_index) {
        int left = middle_index - i - 1;
        while (left >= 0) {
            bins[left] = bins[left + 1] * random_seeds[left];

            left -= 1;
            __syncthreads();
        }
    } 
    else if (i > middle_index) {
        int right = i;
        while (right < NUM_ELEMENTS) {
            bins[right] = bins[right - 1] * random_seeds[right];

            right += 1;
            __syncthreads();
        }
    }
}

__global__ void volatility_optimized(float initial_volatility, float* bins, float* random_seeds) {
    // load random seeds into shared memory and bins
}
