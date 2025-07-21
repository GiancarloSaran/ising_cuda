/* 
 * Parallel 2D Ising model
 * ising_kernels.cu - CUDA kernels for the project. Most of the functions in here either run on the GPU or
 * work with memory addressed in the GPU
 * Giancarlo Saran Gattorno
 * Modern Computing for Physics 2024-25
 */

#include "ising.h"

/**
 * @brief Initializes the values of the look up tables for energy calculation.
 *
 * When validating a Monte Carlo proposal the Metropolis ratio e^{-beta * deltaE} can be calculated directly
 * for every thread, or, since the possible values are finite, using a precomputed lookup table.
 * There are 5 possible values for the interaction energy and 2 values for the external field.
 * This is a common optimization for single threaded calculations 
 * https://stackoverflow.com/questions/74660595/further-optimizing-the-ising-model
 *
 * @param[in] beta Inverse temperature 
 * @param[in] J Coupling strength
 * @param[in] h External field strength
 * @param[out] d_spin_lut DRAM lookup table for interaction energy ratio
 * @param[out] d_field_lut DRAM lookup table for external field ratio
 */

__device__ float d_spin_lut[5];   // exp(-beta*J*{-8,-4,0,4,8})
__device__ float d_field_lut[2];  // exp(-beta*h*{-2,+2})

void init_luts(float beta, float J, float h) {
    float h_spin_lut[5];
    int neighbor_sums[5] = {-8, -4, 0, 4, 8};
    for (int i = 0; i < 5; i++) {
        h_spin_lut[i] = expf(-beta * J * neighbor_sums[i]);
    }
    float h_field_lut[2];
    h_field_lut[0] = expf(2 * beta * h);
    h_field_lut[1] = expf(-2 * beta * h);
    //Move to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_spin_lut, h_spin_lut, sizeof(h_spin_lut)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_field_lut, h_field_lut, sizeof(h_field_lut)));
}


/**
 * @brief Initialize RNG states
 *
 * @param[out] states   Pointer to array of curandState objects to initialize
 * @param[in]  seed     Base random seed for reproducibility
 * @param[in]  n        Number of states to initialize (lattice_size/2 for each color)
 * @param[in]  black_color Color (black if true)
 */

__global__ void init_rng(curandState *states, unsigned long seed, int n, bool black_color) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long color_seed = seed + (black_color ? 123456ULL : 0ULL); // Different seed for two colors
    if (idx < n) {
        curand_init(color_seed, idx, 0, &states[idx]);
    }
}

/**
 * @brief Initialize spin lattice
 *
 * Initialize a single color sublattice of the 2D Ising model with spins set to +1 (up) or -1 (down). 
 * Each thread handles one lattice site, using thread-local random number generation to determine spin orientation 
 * based on the specified probability.
 *
 * The initialization supports both hot start (random spins) and cold start 
 * (ordered spins)
 *
 * @param[out] color        Pointer to device memory for one color sublattice
 * @param[in]  globalStates Pointer to initialized curandState array for random number generation
 * @param[in]  up_prob      Probability of spin being +1 (0.0 to 1.0)
 * @param[in]  nrow         Number of rows in the sublattice
 * @param[in]  ncol         Number of columns in the sublattice
 *
 * @note signed char is the smallest memory type with normal arithmetic (1 byte).
 *       This is the highest compression achievable without having to optimize bitwise operations.
 */
 
 __global__ void init_lattice(signed char *color, curandState *globalStates, float up_prob, int nrow, int ncol){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / ncol;
    int col = idx % ncol;
    if (row >= nrow || col >= ncol) return;
    
    float xi = curand_uniform(&globalStates[idx]);
    color[idx] = (xi < up_prob) ? 1 : -1;
}

// Access LUT for spin interaction
__device__ __forceinline__ float get_spin_factor(int neighbors_sum) {
    int idx = (neighbors_sum + 4) / 2;  // Maps {-4,-2,0,2,4} -> {0,1,2,3,4}
    return d_spin_lut[idx];
}

// Access LUT for external field interaction
__device__ __forceinline__ float get_field_factor(signed char spin) {
    int idx = (spin < 0) ? 0 : 1; // Maps {-1, 1} -> {0, 1}
    return d_field_lut[idx];
}

// Compute sum of nearest neighbor spins for a given site
__device__ signed char get_neighbor_spins(signed char *op_color, int row, int col, 
                                         int nrow, int ncol, bool black_color) {
    int south = (row + 1 < nrow) ? row + 1 : 0;
    int north = (row - 1 >= 0) ? row - 1: nrow - 1;
    int east = (col + 1 < ncol) ? col + 1 : 0;
    int west = (col - 1 >= 0) ? col - 1: ncol - 1;
    
    // Due to the geometry of the checkerboard and row major structure of coalesced memory the neighbors in the opposite color 
    // array are: (1) north, (2) south, (3) same index and (4) east or west alternating by color and row
    int alt_col;
    if (black_color) {
        alt_col = (row % 2) ? east : west; 
    }
    else {
        alt_col = (row % 2) ? west : east;
    }
    
    signed char s_neigh = op_color[north * ncol + col] +
                          op_color[row * ncol + col] + 
                          op_color[south * ncol + col] +
                          op_color[row * ncol + alt_col];
    
    return s_neigh;
}

/**
 * @brief State update for a single color using Metropolis algorithm
 *
 * The main realization for this algorithm is that due to the nearest neighbor interactions of the
 * Ising model the lattice can be divided in a way that reduces the problem of sampling 
 * NxN state updates into two embarassingly parallel tasks, that is like a checkerboard. 
 * Since states of one color only interact with neighbors of the opposite color we can keep one half fixed while we
 * update the other in parallel.
 *
 * @param[in] black_color Color to update (black if true)
 * @param[in] use_lut Metropolis ratio calculation method (lut if true, direct if false)
 * @param     color Pointer to the half checkerboard array to update
 * @param[in] op_color Pointer to the opposite half checkerboard array
 * @param[in] states States of the RNG
 * @param[in] nrow Number of rows
 * @param[in] ncol Number of columns (of the half-board)
 * @param[in] beta Inverse temperature
 * @param[in] J Spin interaction strength
 * @param[in] h External field strength
 */

__global__ void color_update(signed char *color, signed char *op_color,
                            curandState *states, int nrow, int ncol, 
                            float beta, float J, float h, bool use_lut, bool black_color) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = idx / ncol; //lattice row
    const int col = idx % ncol; //lattice column
    if (row >= nrow || col >= ncol) return; //avoid overflow

    signed char s_neigh = get_neighbor_spins(op_color, row, col, nrow, ncol, black_color);    
    signed char s_ij = color[idx];
    float deltaE;
    float metropolis_ratio;
    float xi = curand_uniform(&states[idx]);

    if (use_lut){
        // Use lookup tables
        metropolis_ratio = get_field_factor(s_ij) * get_spin_factor(s_neigh * s_ij);
    }
    else {
        // Direct computation of energy difference and Boltzmann factor
        deltaE = 2 * J * s_ij * s_neigh + 2 * h * s_ij;
        metropolis_ratio = expf(-beta * deltaE);
    }

    // Metropolis acceptance criterion
    if (deltaE <= 0.0f || xi < metropolis_ratio){
        //Accepted MC move, flip the spin
        color[idx] = -s_ij;
    }
}

/**
 * @brief Parallel shared memory reduction kernel
 *
 * This kernel performs the first step of a parallel reduction. That is portion an array
 * into blocks where fast SMP shared memory is used to store intermediate results. This operation is identical
 * for magnetization and energy calculations. What changes is the values we reduce. For the magnetization
 * what we reduce is simply the local spin to get the magnetization of a single color. Then the total can be obtained
 * by simply performing the reduction on both colors and summing the results.
 * For (interaction) energy we have to calculate the product of the local spin and the total neighbor spin at each (single color) site
 * Since energy is 2-body we don't need to perform it on both colors. The additional external field contribution
 * is reutilized from the magnetization reduction. 
 *
 * @param[in] black_color Color on which to perform the reduction (black if true, white if false)
 * @param[in] energy_func Function of the states to reduce (local energy if true, local spin if false)
 * @param[in] color Pointer to the half checkerboard array of which to perform the reduction
 * @param[in] op_color Pointer to the opposite half checkerboard array
 * @param[in] nrow Number of rows
 * @param[in] ncol Number of columns
 * @param[out] blockF Partial reduction results per block
 */

// Parallel reduction kernel for magnetization and energy calculation
__global__ void block_sum(signed char *color, signed char *op_color, 
                         float *blockF, int nrow, int ncol, bool black_color, bool energy_func) {

    int bdim = blockDim.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int start_idx = 2 * bdim * bx;

    __shared__ float sharedF[2 * BLOCK_SIZE]; //We process 2*threads_per_block indices of the array
    
    // We have to initialize to zero so that excess addresses don't influence the reduction
    sharedF[tx] = 0.0f;
    sharedF[tx + bdim] = 0.0f;

    // Indexing of first reduction step
    int idx1 = start_idx + tx;
    int idx2 = start_idx + tx + bdim;
    int row1 = idx1 / ncol;
    int col1 = idx1 % ncol;
    int row2 = idx2 / ncol;
    int col2 = idx2 % ncol;

    // First element
    if (row1 < nrow && col1 < ncol) {
        sharedF[tx] = (float)color[idx1];
        if (energy_func) {
            sharedF[tx] *= (float)get_neighbor_spins(op_color, row1, col1, nrow, ncol, black_color);
        }
    }

    // Strided element
    if (row2 < nrow && col2 < ncol) {
        sharedF[tx + bdim] = (float)color[idx2];
        if (energy_func) {
            sharedF[tx + bdim] *= (float)get_neighbor_spins(op_color, row2, col2, nrow, ncol, black_color);
        }
    }

    __syncthreads(); // Sync the accesses

    // Loop summation with halving strides
    for (int stride = bdim; stride > 0; stride >>= 1) {
        if (tx < stride) {
            sharedF[tx] += sharedF[tx + stride];
        }
        __syncthreads(); //Wait for reduction step to end before accessing other addresses
    }
    
    // Write block result to global memory
    if (tx == 0) {
        blockF[blockIdx.x] = sharedF[0]; 
    }
}