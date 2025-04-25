#include "kernel.cuh"

#define NUM_ELEMENTS 4096

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

__global__ void volatility_optimized(float initial_volatility, float* bins, float* random_seeds) {
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
                summation = (summation * random_seeds[indexing_left]);
                indexing_left -= 1;
            }
        }

        bins[i] = summation * random_seeds[i];
    }
    else if (i > middle_index) {
        int indexing_right = middle_index + 1;
        float summation = initial_volatility;

        while (indexing_right < i) {
            if (summation < 0.0001) {
                break; // end doing trivial computations & conditional checks and just write the summation to all remaining bins
            }
            else {
                summation = (summation * random_seeds[indexing_right]);
                indexing_right += 1;
            }
        }

        bins[i] = summation * random_seeds[i];
    }
    
}
