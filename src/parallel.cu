/* 
 * Parallel 2D Ising model
 * parallel.cu - Main source file. Implements the MCMC loop defined in
 * ising_kernels.cu and deals with memory allocation and cleanup
 * Giancarlo Saran Gattorno
 * Modern Computing for Physics 2024-25
 */

#include "ising.h"

curandState *d_black_states, *d_white_states;

/**
 * @brief Compute energy and magnetization of an Ising lattice
 * The magnetization can be calculated by launching a parallel reduction of black spins, one of
 * white spins and summing them and normalizing by size of the latitce. 
 * The energy is the sum of interaction energy and the total spin multiplied by h
 *
 * @param[in] d_black Black sites spin 
 * @param[in] d_white White sites spin
 * @param[in] dM_block_results Block reduction results for magnetization
 * @param[in] dE_block_results Block reduction results for interaction energy
 * @param[in] nrow
 * @param[in] ncol
 * @param[in] J Coupling strength
 * @param[in] h External field strength
 * @param[out] magnetization 
 * @param[out] energy
 */

void compute_observables(signed char *d_black, signed char *d_white, 
                           float *dM_block_results, float *dE_block_results,
                           int nrow, int ncol, float J, float h, 
                           float *magnetization, float *energy) {
    int n_elements = nrow * ncol/2;
    int num_blocks = (n_elements + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE);
    
    // black magnetization
    block_sum<<<num_blocks, BLOCK_SIZE>>>(d_black, d_white, dM_block_results, nrow, ncol/2, true, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    float black_sum = complete_reduction(dM_block_results, num_blocks);
    
    // white magnetization 
    block_sum<<<num_blocks, BLOCK_SIZE>>>(d_white, d_black, dM_block_results, nrow, ncol/2, false, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    float white_sum = complete_reduction(dM_block_results, num_blocks);    
    float tot_spin = black_sum + white_sum;
    
    // interaction energy (one color to avoid double counting)
    block_sum<<<num_blocks, BLOCK_SIZE>>>(d_black, d_white, dE_block_results, nrow, ncol/2, true, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float interaction_energy = complete_reduction(dE_block_results, num_blocks);

    *energy = -J * interaction_energy - h * tot_spin;
    *magnetization = tot_spin / (nrow * ncol);
}

/**
 * @brief Monte Carlo step
 *
 * A parallel Ising Monte carlo step is composed of two color updates, with device synchronization inbetween
 *
 * @param     d_black Black sites spin 
 * @param     d_white White sites spin
 * @param[in] nrow
 * @param[in] ncol
 * @param[in] beta Inverse temperature
 * @param[in] J Coupling strength
 * @param[in] h External field strength
 * @param[in] use_lut Metropolis ratio calculation method (lut if true, direct if false)
 */

void MCMC_step(signed char *d_black, signed char *d_white, 
               int nrow, int ncol, float beta, float J, float h, bool use_lut) {
    
    int n_elements = nrow * ncol/2;
    int num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Update black sites
    color_update<<<num_blocks, BLOCK_SIZE>>>(
        d_black, d_white, d_black_states, nrow, ncol/2, beta, J, h, use_lut, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Update white sites
    color_update<<<num_blocks, BLOCK_SIZE>>>(
        d_white, d_black, d_white_states, nrow, ncol/2, beta, J, h, use_lut, false);
    CUDA_CHECK(cudaDeviceSynchronize());
}


int main(int argc, char *argv[]) {

    /*
        ################################
        #  PARAMETER PARSING FROM CLI  #
        ################################
    
    */

    int N = 32, n_steps = 500;
    float J = 1.0f, T = 1.5f, h = 0.0f;
    unsigned long seed = 13072025;
    bool hot_start = false;
    bool use_lut = true;
    int block_size = BLOCK_SIZE; //Start with default block size (128)
    int print_freq = 0; //Print every result 10 times per chain
    bool disable_physics_save = true; //Don't save MCMC samples, performance benchmark mode
    int perf_freq = 200; //Calculate performance results every 200 steps

    // Command line argument parsing
    static struct option long_options[] = {
        {"size", required_argument, 0, 'N'},
        {"steps", required_argument, 0, 's'},
        {"seed", required_argument, 0, 1000},
        {"hot", no_argument, 0, 1001},
        {"no-lut", no_argument, 0, 1002},
        {"block-size", required_argument, 0, 1003},
        {"print-freq", required_argument, 0, 1004},
        {"help", no_argument, 0, 1005},
        {"disable-physics", no_argument, 0, 1006},
        {"perf-freq", required_argument, 0, 1007},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "N:s:J:T:h:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'N':
                N = atoi(optarg);
                break;
            case 's':
                n_steps = atoi(optarg);
                break;
            case 'J':
                J = atof(optarg);
                break;
            case 'T':
                T = atof(optarg);
                break;
            case 'h':
                h = atof(optarg);
                break;
            case 1000: // seed
                seed = strtoul(optarg, NULL, 10);
                break;
            case 1001: // hot
                hot_start = true;
                break;
            case 1002: // no-lut
                use_lut = false;
                break;
            case 1003: // block-size
                block_size = atoi(optarg);
                break;
            case 1004: // print-freq
                print_freq = atoi(optarg);
                break;
            case 1005: // help
                print_usage(argv[0]);
                return 0;
            case 1006: // disable-physics
                disable_physics_save = true;
                break;
            case 1007: // perf-freq
                perf_freq = atoi(optarg);
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    if (N <= 0 || n_steps <= 0 || T <= 0.0f) {
        fprintf(stderr, "Error: N, steps, J, T, and h are required parameters\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (print_freq == 0) {
        print_freq = n_steps / 10;
        if (print_freq == 0) print_freq = 1;
    }
    
    // Calculate dependent parameters
    float beta = 1.0f / T;
    int nrow = N, ncol = N;
    int n_elements = nrow * ncol;
    
    printf("2D Ising Model CUDA Simulation\n");
    printf("===============================\n");
    printf("Lattice size: %dx%d\n", nrow, ncol);
    printf("MCMC steps: %d\n", n_steps);
    printf("Temperature: %.6f\n", T);
    printf("Interaction J: %.6f\n", J);
    printf("Magnetic field h: %.6f\n", h);
    printf("Random seed: %lu\n", seed);
    printf("Start type: %s\n", hot_start ? "hot" : "cold");
    printf("Lookup tables: %s\n", use_lut ? "enabled" : "disabled");
    printf("Block size: %d\n", block_size);
    printf("\n");
    
    /*
        #######################
        #  MEMORY ALLOCATION  #
        #######################
    
    */

    signed char *d_black, *d_white;
    float *dM_block_results;
    float *dE_block_results;
    
    CUDA_CHECK(cudaMalloc(&d_black, (n_elements / 2) * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_white, (n_elements / 2) * sizeof(signed char)));
    
    int max_blocks = (n_elements/2 + 2*block_size - 1) / (2*block_size);
    int rng_blocks = (n_elements/2 + block_size - 1) / block_size;
    
    CUDA_CHECK(cudaMalloc(&dM_block_results, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dE_block_results, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_black_states, (n_elements / 2) * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_white_states, (n_elements / 2) * sizeof(curandState)));
    
    /*
        ##########################
        #  SYSTEM INITIALIZATION #
        ##########################
    */

    // Initialize black RNGs
    init_rng<<<rng_blocks, block_size>>>(d_black_states, seed, n_elements/2, true);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize white RNGs 
    init_rng<<<rng_blocks, block_size>>>(d_white_states, seed, n_elements/2, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize lookup tables
    if (use_lut) {
        init_luts(beta, J, h);
    }
    
    // Initialize lattice
    float up_prob = hot_start ? 0.5f : 0.8f;  // Cold start: majority of spins up
    int init_blocks = (n_elements/2 + block_size - 1) / block_size;
    init_lattice<<<init_blocks, block_size>>>(d_black, d_black_states, up_prob, nrow, ncol/2);
    CUDA_CHECK(cudaDeviceSynchronize());
    init_lattice<<<init_blocks, block_size>>>(d_white, d_white_states, up_prob, nrow, ncol/2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initial measurements
    float initial_magnetization;
    float initial_energy;
    compute_observables(d_black, d_white, dM_block_results, dE_block_results, nrow, ncol, J, h,
                        &initial_magnetization, &initial_energy);
    printf("Initial state:\n");
    printf("  Magnetization: %.6f\n", initial_magnetization);
    printf("  Energy: %.6f\n", initial_energy);
    printf("\nStarting MCMC simulation...\n");
    
    clock_t start_time = clock();
    
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "../data/ising_T%.2f_N%d", T, N);
    
    //Initialize savers
    DataSaver* saver = nullptr;
    if (!disable_physics_save){
        DataSaver* saver = init_data_saver(output_filename, 100);  // Save states every 100 steps
    }
    PerformanceMetrics* perf_metrics = init_performance_metrics(output_filename, N, T, J, h, 
                                                            seed, hot_start, use_lut, block_size);

    int mcmc_blocks = (n_elements/2 + block_size - 1) / block_size;
    int max_obs_blocks = (n_elements/2 + 2*block_size - 1) / (2*block_size);

    update_gpu_parameters(perf_metrics, block_size, mcmc_blocks, max_obs_blocks);

    // CUDA events for timing
    cudaEvent_t mcmc_start, mcmc_stop, obs_start, obs_stop;
    CUDA_CHECK(cudaEventCreate(&mcmc_start));
    CUDA_CHECK(cudaEventCreate(&mcmc_stop));
    CUDA_CHECK(cudaEventCreate(&obs_start));
    CUDA_CHECK(cudaEventCreate(&obs_stop));

    // Main MCMC loop
    for (int step = 0; step < n_steps; step++) {
        // Time MCMC step
        CUDA_CHECK(cudaEventRecord(mcmc_start));
        MCMC_step(d_black, d_white, nrow, ncol, beta, J, h, use_lut);
        CUDA_CHECK(cudaEventRecord(mcmc_stop));
        CUDA_CHECK(cudaEventSynchronize(mcmc_stop));
        
        float mcmc_time;
        CUDA_CHECK(cudaEventElapsedTime(&mcmc_time, mcmc_start, mcmc_stop));
        
        CUDA_CHECK(cudaEventRecord(obs_start));
        float magnetization, energy;
        compute_observables(d_black, d_white, dM_block_results, dE_block_results, nrow, ncol, J, h,
                            &magnetization, &energy);
        CUDA_CHECK(cudaEventRecord(obs_stop));
        CUDA_CHECK(cudaEventSynchronize(obs_stop));
        
        float obs_time;
        CUDA_CHECK(cudaEventElapsedTime(&obs_time, obs_start, obs_stop));
        
        // Calculate performance metrics
        perf_metrics->mcmc_time_ms = mcmc_time;
        perf_metrics->observables_time_ms = obs_time;
        
        if (saver != nullptr) {
            save_observables(saver, step, magnetization, energy);
            save_lattice_state(saver, step, d_black, d_white, nrow, ncol);
        }

        if (step % perf_freq == 0) {
            save_performance_metrics(perf_metrics, step);
        }
    }

    /*
    ###################
    #  MEMORY CLEANUP #
    ###################
    */
    if (saver != nullptr) {
        cleanup_data_saver(saver);

    }
    cleanup_performance_metrics(perf_metrics);
    CUDA_CHECK(cudaEventDestroy(mcmc_start));
    CUDA_CHECK(cudaEventDestroy(mcmc_stop));
    CUDA_CHECK(cudaEventDestroy(obs_start));
    CUDA_CHECK(cudaEventDestroy(obs_stop));

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    // Final measurements
    float final_magnetization;
    float final_energy;
    compute_observables(d_black, d_white, dM_block_results, dE_block_results, nrow, ncol, J, h,
                        &final_magnetization, &final_energy);
    
    printf("\nSimulation completed!\n");
    printf("Final state:\n");
    printf("  Magnetization: %.6f\n", final_magnetization);
    printf("  Energy: %.6f\n", final_energy);
    printf("  Elapsed time: %.3f seconds\n", elapsed_time);
    printf("  Performance: %.2f steps/second\n", n_steps / elapsed_time);
    printf("  Throughput: %.2f Mspins/second\n", (n_steps * n_elements) / (elapsed_time * 1e6));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_black));
    CUDA_CHECK(cudaFree(d_white));
    CUDA_CHECK(cudaFree(dE_block_results));
    CUDA_CHECK(cudaFree(dM_block_results));
    CUDA_CHECK(cudaFree(d_black_states));
    CUDA_CHECK(cudaFree(d_white_states));
    
    return 0;
}