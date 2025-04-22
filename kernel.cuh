#pragma once

__global__ void volatility_naive(float initial_volatility, float* bins, float* random_seeds);
__global__ void volatility_optimized(float initial_volatility, float* bins, float* random_seeds);
