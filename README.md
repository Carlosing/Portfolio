# GPU Matrix Multiplication and U-Net Implementation

This repository showcases two main projects demonstrating proficiency in Rust and Python for high-performance computing and deep learning. These projects serve as a portfolio for job applications in software development, particularly in GPU programming and machine learning.

## Projects

### 1. GPU-Accelerated Matrix Multiplication (Rust + CUDA)
This project implements matrix multiplication on the GPU using Rust and CUDA. The program initializes CUDA, allocates memory on the device, and performs matrix multiplication using a custom CUDA kernel. It also benchmarks the performance of different matrix sizes, measuring memory transfer time, computation time, and floating-point operations per second (FLOP/s). The benchmarking results are saved in a CSV file for further analysis.

#### Features:
- Utilizes `rustacuda` for seamless CUDA integration in Rust.
- Implements optimized GPU-accelerated matrix multiplication.
- Measures and records key performance metrics such as memory transfer time, computation time, and FLOP/s.
- Saves benchmarking results in a structured CSV format for analysis.

#### Usage:
To compile and run the Rust program:
```
cargo run --release
```
This will execute the benchmarking tests and generate a CSV file with results.

---

### 2. U-Net Implementation (Jupyter Notebook, Python)
This project implements a U-Net model, a convolutional neural network architecture widely used for image segmentation tasks. The Jupyter Notebook guides users through the entire workflow, from data preprocessing and augmentation to model training and evaluation. The U-Net model is designed to segment images effectively by capturing both spatial and contextual information through its encoder-decoder structure.

#### Features:
- Implements U-Net architecture for high-accuracy image segmentation.
- Performs data preprocessing, including resizing, normalization, and augmentation.
- Trains the model using TensorFlow/Keras with visualization of results.
- Evaluates the model performance and provides qualitative insights through segmented image outputs.

#### Usage:
Open the Jupyter Notebook:
```
jupyter notebook U_net.ipynb
```
Run the cells sequentially to preprocess the dataset, train the model, and visualize the results.

---

## Repository Structure
```
/
├── src/                   # Rust source code for GPU Matrix Multiplication
│   ├── main.rs            # Main Rust program
│   ├── resources/         # Contains CUDA PTX file for matrix multiplication
├── U_net.ipynb            # Jupyter Notebook with U-Net implementation
├── Cargo.toml             # Rust project dependencies
├── benchmark_results.csv  # Output from Rust benchmarking
└── README.md              # Documentation
```

## Contact
If you have any questions or suggestions, feel free to reach out. This repository is a demonstration of skills in GPU programming and deep learning, aimed at potential employers.

