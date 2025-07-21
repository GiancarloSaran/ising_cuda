/* 
 * Parallel 2D Ising model
 * ising.h - Header file for parallel code
 * Giancarlo Saran Gattorno
 * Modern Computing for Physics 2024-25
 */

#ifndef ISING_H
#define ISING_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <string.h>
#include <unistd.h>

// Constants
#define BLOCK_SIZE 128
#define MAX_FILENAME 256
#define tau_c 2.269185f

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Global device pointers
extern curandState *d_black_states, *d_white_states;

// Data saver structure
typedef struct {
    FILE *magnetization_file;
    FILE *energy_file;
    FILE *states_file;
    char *states_filename;
    int save_frequency;
} DataSaver;

/*
    ##########################
    #  FUNCTION DECLARATIONS #
    ##########################
*/


// From ising_kernels.cu - global kernels
__global__ void init_rng(curandState *states, unsigned long seed, int n, bool black_color);
__global__ void init_lattice(signed char *color, curandState *globalStates,
                            float up_prob, int nrow, int ncol);
__global__ void color_update(signed char *color, signed char *op_color, curandState *states, 
                           int nrow, int ncol, float beta, float J, float h, bool use_lut, bool black_color);
__global__ void block_sum(signed char *color, signed char *op_color, float *blockF, 
                         int nrow, int ncol, bool black_color, bool energy_func);

// From ising_kernels.cu - device kernels (for intermediate calculations on the GPU)
__device__ float get_spin_factor(int neighbors_sum);
__device__ float get_field_factor(signed char spin);
__device__ signed char get_neighbor_spins(signed char *op_color, int row, int col, 
                                            int nrow, int ncol, bool black_color);

// From ising_utils.cu - CPU utilities
void init_luts(float beta, float J, float h);
float complete_reduction(float *d_blockResults, int num_blocks);
void print_usage(const char* program_name);

// Data saving
DataSaver* init_data_saver(const char* base_filename, int state_freq);
void save_observables(DataSaver* saver, int step, float magnetization, float energy);
void save_lattice_state(DataSaver* saver, int step, signed char* d_black, signed char* d_white, int nrow, int ncol);
void cleanup_data_saver(DataSaver* saver);

// From parallel.cu - main simulation functions
void compute_observables(signed char *d_black, signed char *d_white, 
                        float *dM_block_results, float *dE_block_results,
                        int nrow, int ncol, float J, float h, 
                        float *magnetization, float *energy);

void MCMC_step(signed char *d_black, signed char *d_white, 
               int nrow, int ncol, float beta, float J, float h, bool use_lut);

// Performance metrics structure with all parameters
typedef struct {
    FILE *performance_file;
    // Timing metrics
    float mcmc_time_ms;
    float observables_time_ms;
    // Physics parameters
    int lattice_size;
    float temperature;
    float interaction_J;
    float magnetic_field_h;
    // Simulation parameters
    unsigned long random_seed;
    bool hot_start;
    bool use_lut;
    // GPU parameters
    int block_size;
    int threads_per_block;
    int blocks_per_grid_mcmc;
    int blocks_per_grid_obs;
} PerformanceMetrics;

// From ising_utils.cu - Performance measurement utilities
PerformanceMetrics* init_performance_metrics(const char* base_filename, int N, float T, float J, float h, 
                                           unsigned long seed, bool hot_start, bool use_lut, int block_size);
void update_gpu_parameters(PerformanceMetrics* metrics, int threads_per_block, int blocks_mcmc, int blocks_obs);
void save_performance_metrics(PerformanceMetrics* metrics, int step);
void cleanup_performance_metrics(PerformanceMetrics* metrics);
#endif // ISING_H