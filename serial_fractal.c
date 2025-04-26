#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//#define NUM_ELEMENTS 131072
#define NUM_ELEMENTS 4096

#define MAX_RANDOM_SEED 0.995
#define MIN_RANDOM_SEED 0.99

void generate_fractal(float initial_volatility, float* bins, float* random_seeds);

void generate_seeds(float* random_bins, float min, float max);

void print_fractal(float* bins);

int main(void) {
    
    clock_t start, end;
    double cpu_time;
    
    start = clock();
   
    float initial_volatility = 1000000.0f;

    float* rand_seeds; 
    float* fractal;

    fractal = (float*)malloc(NUM_ELEMENTS * sizeof(float));
    rand_seeds = (float*)malloc(NUM_ELEMENTS * sizeof(float));

    generate_seeds(rand_seeds, MIN_RANDOM_SEED, MAX_RANDOM_SEED);

    generate_fractal(initial_volatility, fractal, rand_seeds);
    
    //print_fractal(fractal);

    free(fractal);
    free(rand_seeds);
    
    end = clock();
    cpu_time= ((double)(end - start)) / (double)CLOCKS_PER_SEC;
    printf("CPU execution time: %lf seconds\n", cpu_time);
    

    return 0;
}

void generate_fractal(float initial_volatility, float* bins, float* random_seeds) {
    int middle_index = NUM_ELEMENTS / 2;
    
    bins[middle_index] = initial_volatility;
    // generate left side
    int left_index = middle_index - 1;
    while (left_index >= 0) {
        bins[left_index] = bins[left_index + 1] * random_seeds[left_index];

        left_index -= 1;
    }  

    // generate right side
    int right_index = middle_index + 1;
    while (right_index < NUM_ELEMENTS) {
        bins[right_index] = bins[right_index - 1] * random_seeds[right_index];
        
        right_index += 1;
    }
}

void generate_seeds(float* random_bins, float min, float max) {
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        random_bins[i] = min + ((float)rand() / RAND_MAX) * (max - min);
    }
}

void print_fractal(float* bins) {
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        printf("%d: %f\n", i, bins[i]);
    }
}