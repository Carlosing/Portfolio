# CUDA Matrix Multiplication Benchmark

This repository contains a Rust implementation of matrix multiplication using CUDA for GPU acceleration. The project benchmarks the performance of matrix multiplication on the GPU, measuring memory transfer times, computation times, and FLOP/s (Floating Point Operations per Second).

## Overview

The project consists of two main components:

1. **CUDA Matrix Multiplication**: A Rust function that performs matrix multiplication on the GPU using CUDA. The function initializes CUDA, allocates memory on the GPU, transfers data between the CPU and GPU, and executes the matrix multiplication kernel.

2. **Benchmarking**: A benchmarking function that measures the performance of the matrix multiplication for various matrix sizes. The benchmark records memory transfer times, computation times, and FLOP/s, and writes the results to a CSV file.

## Key Features

- **CUDA Integration**: The project uses the `rustacuda` crate to interface with CUDA, allowing for efficient GPU computation.
- **Random Matrix Generation**: Matrices are filled with random floating-point numbers between 0 and 100 to simulate real-world data.
- **Performance Metrics**: The benchmark captures memory transfer times, computation times, and FLOP/s, providing a comprehensive view of GPU performance.
- **CSV Output**: Results are saved in a CSV file (`benchmark_results.csv`), making it easy to analyze and visualize the data.

## Usage

To run the benchmark, ensure you have CUDA installed on your system and the necessary Rust toolchain. Then, clone the repository and run the following command:

```bash
cargo run --release
```

This will execute the benchmark for a set of predefined matrix sizes and save the results to `benchmark_results.csv`.

## Matrix Sizes

The benchmark tests both square and rectangular matrices:

- **Square Matrices**: Sizes include 2x2, 8x8, 16x16, 32x32, 64x64, 128x128, 512x512, and 1024x1024.
- **Rectangular Matrices**: Sizes include (2x8), (8x16), (16x32), (32x64), (64x128), (128x512), and (512x1024).

## Results

The benchmark results include the following metrics for each matrix size:

- **Memory Usage (MB)**: The theoretical memory usage for the matrices.
- **Average Memory Transfer Time (s)**: The average time taken to transfer data between the CPU and GPU.
- **Average Computation Time (s)**: The average time taken for the GPU to perform the matrix multiplication.
- **Average Execution Time (s)**: The total time taken for each benchmark sample.
- **FLOP/s**: The number of floating-point operations per second achieved by the GPU.

## Dependencies

- **rustacuda**: A Rust crate for CUDA programming.
- **rand**: A Rust crate for generating random numbers.
- **csv**: A Rust crate for writing CSV files.

## Future Work

- **Optimization**: Explore further optimizations for the CUDA kernel to improve performance.
- **Larger Matrices**: Extend the benchmark to include larger matrix sizes.
- **Visualization**: Create visualizations of the benchmark results using tools like Python's Matplotlib or Seaborn.
-  **Compare sparse vs dense matrices**
-  **Implement Strassens method**: Implement efficient multiplication parallelized algorithm.
-  **Compare with cuBLAS**

