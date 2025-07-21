#!/bin/bash

# Simple comprehensive test script for parallel.cu
# Focuses on edge cases to clearly demonstrate phase transitions
# Run with: ./test.sh > ../logs/test.log 2>&1

echo "===== Ising Model CUDA Test ====="
echo "Date: $(date)"
echo "Hardware: Jetson Nano (Tegra X1)"
echo "Critical Temperature: T_c = 2.269185"
echo ""

# Compile if needed
if [ ! -f "../bin/parallel" ]; then
    echo "Compiling parallel.cu..."
    nvcc -O3 -arch=sm_53 -maxrregcount=32 --use_fast_math -o ../bin/parallel parallel.cu -lcurand
    if [ $? -ne 0 ]; then
        echo "COMPILATION FAILED!"
        exit 1
    fi
    echo "Compilation successful!"
fi

echo "System Status:"
free -h
echo ""

# Test 1: Very Low Temperature (Strong Order)
echo "=== TEST 1: Very Low Temperature (T=0.5) ==="
echo "Expected: High magnetization (>0.9), strong ferromagnetic order"
../bin/parallel -N 48 -s 3000 -J 1.0 -T 0.5 -h 0.0 --hot --block-size 128 --seed 1001
echo ""

# Test 2: Very High Temperature (Complete Disorder)
echo "=== TEST 2: Very High Temperature (T=10.0) ==="
echo "Expected: Low magnetization (<0.1), random spins"
../bin/parallel -N 48 -s 3000 -J 1.0 -T 10.0 -h 0.0 --hot --block-size 128 --seed 1002
echo ""

# Test 3: Strong External Field (Overcomes thermal disorder)
echo "=== TEST 3: Strong External Field (h=2.0, T=3.0) ==="
echo "Expected: High positive magnetization (>0.7), field-induced order"
../bin/parallel -N 48 -s 3000 -J 1.0 -T 3.0 -h 2.0 --hot --block-size 128 --seed 1003
echo ""

# Test 4: Antiferromagnetic Interactions
echo "=== TEST 4: Antiferromagnetic Interactions (J=-1.0, T=1.0) ==="
echo "Expected: Low magnetization, frustrated interactions"
../bin/parallel -N 48 -s 3000 -J -1.0 -T 1.0 -h 0.0 --hot --block-size 128 --seed 1004
echo ""

# Test 5: Exactly at Critical Temperature
echo "=== TEST 5: Exact Critical Temperature (T=2.269185) ==="
echo "Expected: Critical fluctuations, intermediate magnetization"
../bin/parallel -N 64 -s 5000 -J 1.0 -T 2.269185 -h 0.0 --hot --block-size 128 --seed 1005
echo ""

# Test 6: Just Below Critical Temperature  
echo "=== TEST 6: Slightly Below T_c (T=2.1) ==="
echo "Expected: Ordered phase, |M| > 0.3"
../bin/parallel -N 48 -s 3000 -J 1.0 -T 2.1 -h 0.0 --hot --block-size 128 --seed 1006
echo ""

# Test 7: Just Above Critical Temperature
echo "=== TEST 7: Slightly Above T_c (T=2.4) ==="
echo "Expected: Disordered phase, |M| < 0.2"
../bin/parallel -N 48 -s 3000 -J 1.0 -T 2.4 -h 0.0 --hot --block-size 128 --seed 1007
echo ""

# Test 8: Cold Start Below T_c (Equilibration test)
echo "=== TEST 8: Cold Start Below T_c (T=1.5) ==="
echo "Expected: Should maintain high magnetization"
../bin/parallel -N 48 -s 3000 -J 1.0 -T 1.5 -h 0.0 --block-size 128 --seed 1008
echo ""

# Test 9: Weak Field at Critical Point
echo "=== TEST 9: Weak Field at Critical Point (h=0.01, T=2.27) ==="
echo "Expected: Slight bias toward positive magnetization"
../bin/parallel -N 48 -s 4000 -J 1.0 -T 2.27 -h 0.01 --hot --block-size 128 --seed 1009
echo ""

# Test 10: Size Scaling at Critical Temperature
echo "=== TEST 10: Size Scaling at T_c ==="
echo "Testing different lattice sizes at critical temperature"
for N in 32 48 64; do
    echo "--- Lattice Size: ${N}x${N} ---"
    ../bin/parallel -N $N -s 2000 -J 1.0 -T 2.269 -h 0.0 --hot --block-size 128 --seed $((2000+N))
    echo ""
done

# Test 11: Performance Comparison
echo "=== TEST 11: LUT Performance Comparison ==="
echo "With lookup tables:"
../bin/parallel -N 64 -s 1500 -J 1.0 -T 2.269 -h 0.0 --hot --block-size 128 --seed 3001 | grep "Performance:"

echo "Without lookup tables:"
../bin/parallel -N 64 -s 1500 -J 1.0 -T 2.269 -h 0.0 --no-lut --hot --block-size 128 --seed 3001 | grep "Performance:"
echo ""

echo "===== SUMMARY ====="
echo "Expected Results:"
echo "• T=0.5:  |M| > 0.9  (Strong ferromagnetic order)"
echo "• T=10.0: |M| < 0.1  (Complete disorder)"
echo "• h=2.0:  M > 0.7    (Field-induced alignment)"
echo "• J=-1.0: |M| ≈ 0.1  (Antiferromagnetic frustration)"
echo "• T≈T_c:  |M| ≈ 0.1-0.4 (Critical fluctuations)"
echo "• T<T_c:  |M| > 0.3  (Ordered phase)"
echo "• T>T_c:  |M| < 0.2  (Disordered phase)"
echo ""
echo "Performance on Jetson Nano: ~20K-80K spins/second"
echo "Test completed: $(date)"