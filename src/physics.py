import numpy as np
import subprocess
import os

def launch_ising(T, N=512, J=1, h=0, steps=10000, executable_path="../bin/parallel"):
    """
    Launch CUDA Ising model simulation
    
    Args:
        T (float): Temperature
        N (int): Lattice size (NxN)
        J (float): Coupling constant
        h (float): External magnetic field
        steps (int): Number of MCMC steps
        executable_path (str): Path to compiled CUDA executable
    
    Returns:
        int: Return code (0 for success)
    """
    
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
        "-h", str(h)
    ]
    
    print(f"Running: T={T:.3f}, N={N}x{N}, J={J}, h={h}, steps={steps}")
    
    try:
        # Run the simulation
        result = subprocess.run(cmd, check=True)
        print(f"Completed T={T:.3f}")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation at T={T:.3f}: {e}")
        return e.returncode
    except FileNotFoundError:
        print(f"Executable not found: {executable_path}")
        return 1
    except Exception as e:
        print(f"Unexpected error at T={T:.3f}: {e}")
        return 1

# Temperature sweep
T_space = np.linspace(1, 3, 10)

print("Starting Ising model temperature sweep...")
print(f"Temperature range: {T_space[0]:.3f} to {T_space[-1]:.3f}")
print(f"Critical temperature (theoretical): {2/np.log(1 + np.sqrt(2)):.3f}")
print("=" * 50)

for i, T in enumerate(T_space):
    print(f"[{i+1}/{len(T_space)}] ", end="")
    launch_ising(T)

print("=" * 50)
print("Temperature sweep completed!")
print(f"Data files saved in ../data/ directory")