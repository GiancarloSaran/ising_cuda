/* 
 * Parallel 2D Ising model
 * ising_utils.cu - utility functions that run on the CPU. Mostly to deal with the IO of results and
 * the command line interface
 * Giancarlo Saran Gattorno
 * Modern Computing for Physics 2024-25
 */

#include "ising.h"


/**
 * @brief Complete reduction on CPU
 *
 * In the parallel reduction algorithm a first step is done on shared device memory to reduce the results
 * down to block level. Then a final serial reduction of few items is carried out on the host
 *
 * @param[in] d_blockResults Pointer to partial block-wise reduction results (on global device memory) 
 * @param[in] num_blocks size of the global device memory array d_blockResults
 * @param[out] total Final reduction results
 */

float complete_reduction(float *d_blockResults, int num_blocks) {
    //Move to host
    float *h_blockResults = (float*)malloc(num_blocks * sizeof(float));
    if (h_blockResults == NULL){
        printf("Failed to allocate block results\n");
    }
    CUDA_CHECK(cudaMemcpy(h_blockResults, d_blockResults, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    //Final reduction
    float total = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total += h_blockResults[i];
    }
    
    free(h_blockResults);
    return total;
}

// Print usage information for CLI
void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -N, --size <N>        Lattice size (NxN) [required]\n");
    printf("  -s, --steps <steps>   Number of MCMC steps [required]\n");
    printf("  -J <J>                Interaction strength [required]\n");
    printf("  -T <T>                Temperature [required]\n");
    printf("  -h <h>                External magnetic field [required]\n");
    printf("  --seed <seed>         Random seed (default: time-based)\n");
    printf("  --hot                 Hot start (random spins, default: cold start)\n");
    printf("  --no-lut              Disable lookup tables\n");
    printf("  --save-physics        Save MCMC\n");
    printf("  --block-size <size>   CUDA block size (default: 256)\n");
    printf("  --print-freq <freq>   Print frequency (default: steps/10)\n");
    printf("  --help                Show this help message\n");
    printf("\nExample:\n");
    printf("  %s -N 64 -s 10000 -J 1.0 -T 2.5 -h 0.0\n", program_name);
}


// Initialize data saving structure
DataSaver* init_data_saver(const char* base_filename, int state_freq) {
    DataSaver* saver = (DataSaver*)malloc(sizeof(DataSaver));
    char filename[256];

    snprintf(filename, sizeof(filename), "%s_magnetization.csv", base_filename);
    bool file_exists = (access(filename, F_OK) == 0);
    saver->magnetization_file = fopen(filename, "a");
    if (!file_exists) {
        fprintf(saver->magnetization_file, "step,magnetization\n");
    }
    
    snprintf(filename, sizeof(filename), "%s_energy.csv", base_filename);
    file_exists = (access(filename, F_OK) == 0);
    saver->energy_file = fopen(filename, "a");
    if (!file_exists) {
        fprintf(saver->energy_file, "step,energy\n");
    }
    
    // Store state filename template
    saver->states_filename = (char*)malloc(256);
    snprintf(saver->states_filename, 256, "%s_state", base_filename);
    
    saver->save_frequency = state_freq;
    return saver;
}

// Save magnetization and energy
void save_observables(DataSaver* saver, int step, float magnetization, float energy) {
    fprintf(saver->magnetization_file, "%d,%.8f\n", step, magnetization);
    fprintf(saver->energy_file, "%d,%.8f\n", step, energy);
    fflush(saver->magnetization_file); 
    fflush(saver->energy_file);
}

/**
 * @brief Save full lattice state in binary format
 *
 * File format:
 * - Header: nrow (int), ncol (int)
 * - Data: black spins array, white spins array
 *
 * @param[in] saver Pointer to DataSaver structure
 * @param[in] step Current simulation step
 * @param[in] d_black Device pointer to black sublattice
 * @param[in] d_white Device pointer to white sublattice
 * @param[in] nrow Number of lattice rows
 * @param[in] ncol Number of lattice columns
 */

void save_lattice_state(DataSaver* saver, int step, signed char* d_black, signed char* d_white, 
                       int nrow, int ncol) {
    if (step % saver->save_frequency != 0) return;
    
    char filename[512];
    snprintf(filename, sizeof(filename), "%s_%06d.bin", saver->states_filename, step);
    
    FILE* file = fopen(filename, "wb");
    if (!file) return;
    
    // Write header: dimensions
    fwrite(&nrow, sizeof(int), 1, file);
    fwrite(&ncol, sizeof(int), 1, file);
    
    // Copy device data to host
    int n_elements = nrow * ncol / 2;
    signed char* h_black = (signed char*)malloc(n_elements * sizeof(signed char));
    signed char* h_white = (signed char*)malloc(n_elements * sizeof(signed char));
    
    CUDA_CHECK(cudaMemcpy(h_black, d_black, n_elements * sizeof(signed char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_white, d_white, n_elements * sizeof(signed char), cudaMemcpyDeviceToHost));
    
    // Write lattice data
    fwrite(h_black, sizeof(signed char), n_elements, file);
    fwrite(h_white, sizeof(signed char), n_elements, file);
    
    fclose(file);
    free(h_black);
    free(h_white);
}

void cleanup_data_saver(DataSaver* saver) {
    if (saver->magnetization_file) fclose(saver->magnetization_file);
    if (saver->energy_file) fclose(saver->energy_file);
    free(saver->states_filename);
    free(saver);
}

// Performance metrics
PerformanceMetrics* init_performance_metrics(const char* base_filename, int N, float T, float J, float h, 
                                           unsigned long seed, bool hot_start, bool use_lut, int block_size) {
    PerformanceMetrics* metrics = (PerformanceMetrics*)malloc(sizeof(PerformanceMetrics));
    char filename[256];
    
    snprintf(filename, sizeof(filename), "%s_performance.csv", base_filename);
    bool file_exists = (access(filename, F_OK) == 0);
    metrics->performance_file = fopen(filename, "a");
    
    if (!file_exists) {
        fprintf(metrics->performance_file, 
                "step,"
                "mcmc_time_ms,observables_time_ms,"
                "lattice_size,temperature,interaction_J,magnetic_field_h,"
                "random_seed,hot_start,use_lut,"
                "block_size,threads_per_block,blocks_per_grid_mcmc,blocks_per_grid_obs\n");
    }
    
    // Store simulation parameters
    metrics->lattice_size = N;
    metrics->temperature = T;
    metrics->interaction_J = J;
    metrics->magnetic_field_h = h;
    metrics->random_seed = seed;
    metrics->hot_start = hot_start;
    metrics->use_lut = use_lut;
    metrics->block_size = block_size;
    return metrics;
}

void update_gpu_parameters(PerformanceMetrics* metrics, int threads_per_block, int blocks_mcmc, int blocks_obs) {
    metrics->threads_per_block = threads_per_block;
    metrics->blocks_per_grid_mcmc = blocks_mcmc;
    metrics->blocks_per_grid_obs = blocks_obs;
}

void save_performance_metrics(PerformanceMetrics* metrics, int step) {
    //Store in CSV
    fprintf(metrics->performance_file, 
            "%d,"
            "%.6f,%.6f,"
            "%d,%.6f,%.6f,%.6f,"
            "%lu,%d,%d,"
            "%d,%d,%d,%d\n",
            step,
            // Timing
            metrics->mcmc_time_ms, metrics->observables_time_ms,
            // Physical parameters
            metrics->lattice_size, metrics->temperature, metrics->interaction_J, metrics->magnetic_field_h,
            // Simulation parameters
            metrics->random_seed, metrics->hot_start ? 1 : 0, metrics->use_lut ? 1 : 0,
            // GPU parameters
            metrics->block_size, metrics->threads_per_block, 
            metrics->blocks_per_grid_mcmc, metrics->blocks_per_grid_obs);
    fflush(metrics->performance_file);
}

void cleanup_performance_metrics(PerformanceMetrics* metrics) {
    if (metrics->performance_file) fclose(metrics->performance_file);
    free(metrics);
}