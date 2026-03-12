#!/bin/bash
# Quick reference for running accel-sim with OpenMP

ACCEL_SIM_DIR=/home/mmy/work/gpgpu-sim/accel-sim-framework

# Always source the environment first
source $ACCEL_SIM_DIR/gpu-simulator/setup_environment.sh

# Serial execution (baseline)
OMP_NUM_THREADS=1 $ACCEL_SIM_DIR/gpu-simulator/bin/release/accel-sim.out [args...]

# Parallel execution (8 threads)
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread $ACCEL_SIM_DIR/gpu-simulator/bin/release/accel-sim.out [args...]

# Parallel with dynamic scheduling
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_SCHEDULE=dynamic,1 $ACCEL_SIM_DIR/gpu-simulator/bin/release/accel-sim.out [args...]

# Check OpenMP is working
OMP_DISPLAY_ENV=TRUE OMP_NUM_THREADS=4 $ACCEL_SIM_DIR/gpu-simulator/bin/release/accel-sim.out [args...]
