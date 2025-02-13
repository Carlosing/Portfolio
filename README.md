# CUDA Matrix Multiplication Benchmark

This repository contains a Rust implementation of matrix multiplication using CUDA for GPU acceleration. The project benchmarks the performance of matrix multiplication on the GPU, measuring memory transfer times, computation times, and FLOP/s (Floating Point Operations per Second). The results are saved in a CSV file for further analysis.

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

``` bash cargo run --release ```
