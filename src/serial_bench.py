import numpy as np
import subprocess
import os

def launch_ising_serial(N, T=2, J=1, h=0, steps=1000, executable_path="../bin/serial"):
    cmd = [
        executable_path,
        "-N", str(N),
        "-s", str(steps),
        "-J", str(J),
        "-T", str(T),
        "-H", str(h)
    ]
    print(f"Running: N={N}x{N}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    N_space = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    for i, N in enumerate(N_space):
        print(f"[{i+1}/{len(N_space)}] ", end="")
        launch_ising_serial(N=N)

    print("Lattice size sweep completed")
