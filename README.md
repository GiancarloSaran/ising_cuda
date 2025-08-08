# Parallel 2D Ising - Modern Computing for Physics 
![Ising Model](postproc/plots/spin.png)


Monte Carlo simulation of the 2D Ising model implementing both sequential C and parallel CUDA versions. The Glauber (single spin) Metropolis proposal is parallelized with checkerboard decomposition for race-free updates, optimized CUDA kernels with shared memory and coalesced access patterns, and parallel reduction for energy and magnetization calculations.