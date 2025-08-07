import numpy as np
import subprocess
import os

def launch_ising(N=512, block_size=128, T=2, J=1, h=0, steps=1000, use_lut=True, executable_path="../bin/parallel"):
    # Check if executable exists
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        print("Make sure you've compiled the CUDA code with 'make' in the source directory")
        return 1
    
    # Build command
    cmd = [
        executable_path,
        "-N", str(N),
        "-s", str(steps),
        "-J", str(J),
        "-T", str(T),
        "-h", str(h),
        "--block-size", str(block_size)       
    ]
    if not use_lut:
        cmd.append("--no-lut")
    
    print(f"Running: N={N}x{N}, {block_size} threads per block, use_lut = {use_lut}")
    result = subprocess.run(cmd, check=True)
    return result 

# Temperature sweep
N_space = [32, 64, 128, 256, 512, 1024, 2048, 4096]


for i, N in enumerate(N_space):
    launch_ising(N=N)
    launch_ising(N=N, use_lut=False)

print("Lattice size sweep completed")

block_sizes = [32, 64, 128, 256, 512, 1024]
for i, b in enumerate(block_sizes):
    launch_ising(block_size=b)

print("Block size sweep completed")
