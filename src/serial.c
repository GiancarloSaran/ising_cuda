#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
#include <stdbool.h>

/* 
 * Serial 2D Ising model
 * Giancarlo Saran Gattorno
 * Modern Computing for Physics 2024-25
 */
// Compile with gcc -O3 -march=native -flto -o ../bin/serial serial.c -lm

// Default parameters
#define DEFAULT_L 32
#define DEFAULT_STEPS 500
#define DEFAULT_J 1.0
#define DEFAULT_T 2.1
#define DEFAULT_H 0.0
#define DEFAULT_SEED 13072025
#define DEFAULT_HOT_START 0


/*
    ######################
    #  MEMORY ALLOCATION #
    ######################
*/

signed char* alloc_lattice(int L) {
    return (signed char*)malloc(L * L * sizeof(signed char));
}

void free_lattice(signed char *lattice) {
    free(lattice);
}


/*
    #############################
    #  RANDOM NUMBER GENERATORS #
    #############################
*/

double uniform_random() {
    return (double)rand() / RAND_MAX;
}
int randint(int min, int max) {
    return min + rand() % (max - min);
}

/*
    ################
    #  OBSERVABLES #
    ################
*/

double get_energy(signed char *spin, int L, double J, double h) {
    double total_energy = 0.0;
    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            int idx = x * L + y;
            
            int east = x * L + ((y + 1) % L);
            int south = ((x + 1) % L) * L + y;
            
            // interaction energy (half to avoid double count)
            total_energy -= J * spin[idx] * spin[east];
            total_energy -= J * spin[idx] * spin[south];
            
            // magnetic field contribution
            total_energy -= h * spin[idx];
        }
    }
    return total_energy;
}

double get_magnetization(signed char *spin, int L) {
    int total_spin = 0;
    for (int i = 0; i < L * L; i++) {
        total_spin += spin[i];
    }
    return (double)total_spin / (L * L);
}

/*
    ################
    #  MONTE CARLO #
    ################
*/

/**
 * @brief Initialize spin configuration
 * 
 * @param[in] spin Pointer to lattice
 * @param[in] L Size of the lattice 
 * @param[in] hot_start true if random start, false if aligned start
 * @param[in] seed RNG seed
 */

void init_lattice(signed char *spin, int L, int hot_start, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < L * L; i++) {
        if (hot_start) {
            spin[i] = (rand() % 2) * 2 - 1; // Random +1 or -1
        } else {
            spin[i] = 1; // All spins up (cold start)
        }
    }
}

/**
 * @brief Calculate change in energy for Glauber dynamics
 * If σ is the spin flipped and S the sum of neighboring spins
 * ΔE = 2J·σ·S + 2h·σ
 *
 * @param[in] spin Pointer to lattice
 * @param[in] x Horizontal lattice coordinate
 * @param[in] y Vertical lattice coordinate
 * @param[in] L Size of the lattice 
 * @param[in] J Spin-spin interaction strength
 * @param[in] h External field
 */

double get_deltaE(signed char *spin, int x, int y, int L, double J, double h) {
    int idx = x * L + y;
    int east = x * L + ((y + 1) % L);
    int west = x * L + ((y - 1 + L) % L);
    int north = ((x + 1) % L) * L + y;
    int south = ((x - 1 + L) % L) * L + y;
    int S = spin[east] + spin[west] + spin[north] + spin[south];
    double deltaE = 2.0 * J * spin[idx] * S + 2.0 * h * spin[idx];
    return deltaE;
}

/**
 * @brief Single spin MCMC update
 * 
 * @param[in] spin Pointer to lattice
 * @param[in] x Horizontal lattice coordinate
 * @param[in] y Vertical lattice coordinate
 * @param[in] L Size of the lattice 
 * @param[in] J Spin-spin interaction strength
 * @param[in] h External field
 * @param[in] beta inverse temperature
 * @param[in] current_energy
 * @param[in] current_magnetization
 */

int glauber_update(signed char *spin, int x, int y, int L, double J, double h, double beta, 
                        double *current_energy, double *current_magnetization) {
    
    int idx = x * L + y;
    int total_sites = L * L;    
    
    // Metropolis criterion
    double deltaE = get_deltaE(spin, x, y, L, J, h);
    double xi = uniform_random();
    int do_flip = (deltaE <= 0) || (xi < exp(-beta * deltaE));
    
    if (do_flip) {
        signed char old_spin = spin[idx];
        spin[idx] = -old_spin;
        *current_energy += deltaE;
        *current_magnetization += -2.0 * old_spin / total_sites;
        return 1;
    }
    return 0;
}

/**
 * @brief Full MCMC step
 * Every step of the MCMC algorithm all LxL spin flips are attempted
 * 
 * @param spin Pointer to lattice state
 * @param[out] current_magnetization
 * @param[out] current_energy
 * @param[in] L Size of the lattice 
 * @param[in] J Spin-spin interaction strength
 * @param[in] h External field
 * @param[in] beta inverse temperature
 */

void MCMC_sweep(signed char *spin, double *current_energy, double *current_magnetization,
                int L, double J, double h, double beta) {
    for (int x = 0; x < L; x++){
        for (int y = 0; y < L; y++){
            glauber_update(spin, x, y, L, J, h, beta,
                        current_energy, current_magnetization);
        }
    }
}

// Print usage information
void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  -N, --size SIZE         Lattice size (default: %d)\n", DEFAULT_L);
    printf("  -s, --steps STEPS       Number of MCMC steps (default: %d)\n", DEFAULT_STEPS);
    printf("  -J, --coupling J        Coupling strength (default: %.1f)\n", DEFAULT_J);
    printf("  -T, --temperature T     Temperature (default: %.1f)\n", DEFAULT_T);
    printf("  -H, --field H           External magnetic field (default: %.1f)\n", DEFAULT_H);
    printf("      --seed SEED         Random seed (default: time-based)\n");
    printf("      --hot               Hot start (random spins, default)\n");
    printf("      --cold              Cold start (all spins up)\n");
    printf("      --help              Show this help message\n");
}

void save_observables(const char* filename, double *magnetization, double *energy, int steps) {
    bool file_exists = (access(filename, F_OK) == 0);
    FILE *file = fopen(filename, "a");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return;
    }
    
    if (!file_exists) { //Write header
        fprintf(file, "step,magnetization,energy\n");
    }
    
    for (int i = 0; i < steps; i++) {
        fprintf(file, "%d,%.6f,%.6f\n", i, magnetization[i], energy[i]);
    }
    fflush(file);
    fclose(file);
}

// Save performance metrics
void save_performance(const char* filename, int L, double T, double J, double h, 
                     int steps, double *mcmc_time) {
    bool file_exists = (access(filename, F_OK) == 0);
    FILE *file = fopen(filename, "a");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return;
    }

    if (!file_exists) {//Write header
        fprintf(file, "L,T,J,h,steps,mcmc_step_ms\n");
    }
    
    for (int step = 0; step < steps; step++){
        fprintf(file, "%d,%.6f,%.6f,%.6f,%d,%.6f\n", 
            L, T, J, h, step, mcmc_time[step]);
    }
    fflush(file);
    fclose(file);
}

int main(int argc, char *argv[]) {
    // Default parameters
    int L = DEFAULT_L;
    int steps = DEFAULT_STEPS;
    double J = DEFAULT_J;
    double T = DEFAULT_T;
    double h = DEFAULT_H;
    unsigned int seed = DEFAULT_SEED;
    int hot_start = DEFAULT_HOT_START;
    
    // Command line parsing
    static struct option long_options[] = {
        {"size", required_argument, 0, 'N'},
        {"steps", required_argument, 0, 's'},
        {"coupling", required_argument, 0, 'J'},
        {"temperature", required_argument, 0, 'T'},
        {"field", required_argument, 0, 'H'},
        {"seed", required_argument, 0, 1000},
        {"hot", no_argument, 0, 1001},
        {"cold", no_argument, 0, 1002},
        {"help", no_argument, 0, 1003},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "N:s:J:T:H:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'N':
                L = atoi(optarg);
                break;
            case 's':
                steps = atoi(optarg);
                break;
            case 'J':
                J = atof(optarg);
                break;
            case 'T':
                T = atof(optarg);
                break;
            case 'H':
                h = atof(optarg);
                break;
            case 1000: // --seed
                seed = (unsigned int)atoi(optarg);
                break;
            case 1001: // --hot
                hot_start = 1;
                break;
            case 1002: // --cold
                hot_start = 0;
                break;
            case 1003: // --help
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    double beta = 1.0 / T;
    
    printf("2D Ising Model Serial Monte Carlo Simulation\n");
    printf("==============================================\n");
    printf("Lattice size: %dx%d\n", L, L);
    printf("Temperature: %.6f\n", T);
    printf("Coupling J: %.6f\n", J);
    printf("Magnetic field h: %.6f\n", h);
    printf("Steps: %d\n", steps);
    printf("Random seed: %u\n", seed);
    printf("Start type: %s\n", hot_start ? "Hot" : "Cold");
    printf("\n");
    
    // Allocate memory
    signed char *spin = alloc_lattice(L);
    double *mcmc_time = malloc(steps * sizeof(double));
    double *magnetization_data = malloc(steps * sizeof(double));
    double *energy_data = malloc(steps * sizeof(double));
    
    if (!spin || !magnetization_data || !energy_data) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }
    
    init_lattice(spin, L, hot_start, seed);
    
    // MCMC loop
    double energy, magnetization;
    energy = get_energy(spin, L, J, h);
    magnetization = get_magnetization(spin, L);
    for (int t = 0; t < steps; t++){
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        MCMC_sweep(spin, &energy, &magnetization, L, J, h, beta);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 + 
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;
        mcmc_time[t] = time_ms;
        magnetization_data[t] = magnetization;
        energy_data[t] = energy;
    } 
    
    printf("Results:\n");
    printf("--------\n");
    printf("Final magnetization: %.6f\n", magnetization_data[steps-1]);
    printf("Final energy per site: %.6f\n", energy_data[steps-1] / (L * L));
    printf("\n");
    
    //Data saving
    char mag_filename[256], perf_filename[256];
    snprintf(mag_filename, sizeof(mag_filename), "../data/serial_N%d_T%.3f_J%.1f_h%.1f_s%d.csv", 
             L, T, J, h, steps);
    snprintf(perf_filename, sizeof(perf_filename), "../data/serial_performance_N%d_T%.3f.csv", L, T);
    save_observables(mag_filename, magnetization_data, energy_data, steps);
    save_performance(perf_filename, L, T, J, h, steps, mcmc_time);
    
    printf("Data saved:\n");
    printf("- Observables: %s\n", mag_filename);
    printf("- Performance: %s\n", perf_filename);
    
    // Cleanup
    free_lattice(spin);
    free(magnetization_data);
    free(energy_data);
    free(mcmc_time);

    return 0;
}