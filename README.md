# Project Portfolio

This repository contains two key projects showcasing my skills in code development and optimization using Rust and Python for Machine Learning and High-Performance Computing tasks.

## Projects

### 1. Matrix Multiplication on GPU with Rust
**File:** `main.rs`

This project implements an optimized solution for matrix multiplication using the GPU with Rust and the `rustacuda` library. Key features include:

- GPU buffer initialization and management.
- Loading and executing a CUDA kernel for matrix multiplication.
- Measuring memory transfer times and kernel execution time.
- Automatic benchmarking with matrices of varying sizes and generating results in CSV format.

This project demonstrates my ability to work with GPU computing, code optimization, and performance measurement.

### 2. Image Segmentation with U-Net in PyTorch
**File:** `U_net.ipynb`

This notebook implements a U-Net neural network in PyTorch for image segmentation. It includes:

- Definition of the U-Net model with convolutional, pooling, and upsampling layers.
- Training on an image segmentation dataset.
- Model evaluation using performance metrics.
- Visualization of predictions and comparison with ground truth labels.

This project highlights my skills in Deep Learning, convolutional networks, and model implementation in PyTorch.

## Installation and Usage

### Matrix Multiplication on GPU
1. Install Rust dependencies:
   ```sh```
   cargo install rustacuda
