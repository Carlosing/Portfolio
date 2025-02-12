#[macro_use]
extern crate rustacuda;


extern crate rustacuda_derive;
extern crate rustacuda_core;
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
use std::error::Error;
use std::ffi::CString;
use rand::Rng;
use std::time::Instant;
use csv::Writer; // Import the csv crate

// Function to perform matrix multiplication on the GPU
fn cuda_mul(h_a: &Vec<f64>, h_b: &Vec<f64>, rows: usize, cols: usize, size: usize) -> Result<(Vec<f64>, f64, f64), Box<dyn Error>> {
    // Initialize CUDA
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Generate input matrices with random numbers between 0 and 100
    let mut h_c = vec![0.0f64; rows * size];

    // Start measuring memory transfer time
    let start_mem_transfer = Instant::now();

    // Allocate buffers on the GPU
    let mut d_a = DeviceBuffer::from_slice(h_a)?;
    let mut d_b = DeviceBuffer::from_slice(h_b)?;
    let mut d_c = DeviceBuffer::from_slice(&h_c)?;

    // Copy data to the GPU
    d_a.copy_from(h_a)?;
    d_b.copy_from(h_b)?;

    // Measure memory transfer duration
    let mem_transfer_duration = start_mem_transfer.elapsed();

    // Start measuring computation time (GPU computation)
    let start_computation = Instant::now();

    // Load the kernel from the PTX file
    let module_data = CString::new(include_str!("../resources/matrix_mul.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a CUDA stream
    let stream = Stream::new(rustacuda::stream::StreamFlags::DEFAULT, None)?;

    // Convert kernel name to CString
    let kernel_name = CString::new("matrix_mul")?;
    // Get the kernel function from the module
    let func = module.get_function(kernel_name.as_c_str())?;

    // Configure the kernel
    let block_size = 16;
    let grid = ((size as u32 + block_size - 1) / block_size, 
            (rows as u32 + block_size - 1) / block_size, 1);
    let block = (block_size, block_size, 1);

    // Launch the kernel
    unsafe {
        launch!(func<<<grid, block, 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_c.as_device_ptr(),
            rows,
            cols,
            size
        ))?;
    }

    // Wait for the kernel to finish
    stream.synchronize()?;

    // Copy results back to the host
    d_c.copy_to(&mut h_c)?;

    // Measure computation duration
    let computation_duration = start_computation.elapsed();

    // Return the result and timings
    Ok((h_c, mem_transfer_duration.as_secs_f64(), computation_duration.as_secs_f64()))
}

// Function to benchmark the GPU matrix multiplication
fn benchmark_gpu(a: &Vec<f64>, b: &Vec<f64>, n: usize, m: usize, p: usize, sample_size: usize, wtr: &mut Writer<std::fs::File>) -> Result<(), Box<dyn Error>> {
    // Calculate memory usage (in bytes) using f64 (8 bytes per element)
    let memory_usage = (n * m + m * p + n * p) * 8;  // 8 bytes per f64
    let memory_usage_mb = memory_usage as f64 / (1024.0 * 1024.0); // Convert bytes to MB
    println!("Theoretical memory usage for {}x{} * {}x{} matrix: {:.6} MB", n, m, m, p, memory_usage_mb);

    // Calculate the total floating point operations
    let flop_count = n * p * (2 * m - 1);  // Total floating point operations

    // Initialize total times
    let mut total_mem_transfer_time = 0.0;
    let mut total_computation_time = 0.0;
    let mut total_execution_time = 0.0;

    // Perform the benchmark for the given sample size
    for _ in 0..sample_size {
        let mut mem_transfer_time = 0.0;
        let mut computation_time = 0.0;

        // Warm-up the GPU with one iteration (ensures the GPU reaches a steady state)
        let _=cuda_mul(a, b, n, m, p);

        // Start measuring the sample execution time
        let sample_start = Instant::now();

        // Perform matrix multiplication on GPU for this sample
        let (_, mem_transfer, comp_time) = cuda_mul(&a, &b, n, m, p)?;

        // Measure the execution time for this sample
        let execution_time = sample_start.elapsed().as_secs_f64();

        // Accumulate the times for this sample
        mem_transfer_time += mem_transfer;
        computation_time += comp_time;

        // Add the times for this sample to the totals
        total_mem_transfer_time += mem_transfer_time;
        total_computation_time += computation_time;
        total_execution_time += execution_time; // Add the execution time
    }

    // Compute averages over the sample size
    let average_mem_transfer_time = total_mem_transfer_time / sample_size as f64;
    let average_computation_time = total_computation_time / sample_size as f64;
    let average_execution_time = total_execution_time / sample_size as f64;
    let flop_per_sec = flop_count as f64 / average_computation_time;

    // Print the results
    println!(
        "Average Memory Transfer Time (GPU -> CPU + CPU -> GPU) over {} samples: {:.6} seconds",
        sample_size, average_mem_transfer_time
    );
    println!(
        "Average Computation Time (GPU Kernel Execution) over {} samples: {:.6} seconds",
        sample_size, average_computation_time
    );
    println!(
        "Average Execution Time (Total time for each benchmark sample) over {} samples: {:.6} seconds",
        sample_size, average_execution_time
    );
    println!(
        "FLOP/s for {}x{} * {}x{} matrix: {:.6} FLOP/s",
        n, m, m, p, flop_per_sec
    );

    // Write the results to the CSV file
    wtr.write_record(&[
        n.to_string(),
        m.to_string(),
        p.to_string(),
        format!("{:.6}", memory_usage_mb),
        format!("{:.6}", average_mem_transfer_time),
        format!("{:.6}", average_computation_time),
        format!("{:.6}", average_execution_time),
        format!("{:.6}", flop_per_sec),
    ])?;

    Ok(())
}

// Main function to perform the benchmarking
fn main() -> Result<(), Box<dyn Error>> {
    // Define the set of square matrix sizes
    let square_sizes = vec![2, 8, 16, 32, 64, 128, 512, 1024]; // Square matrices sizes

    // Define the set of rectangular matrix sizes in pairs (n x m) * (m x n) only
    let rectangular_sizes = vec![(2, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 512), (512, 1024)];

    // Sample size (number of samples or measurements)
    let sample_size = 100;

    // Create a random number generator
    let mut rng = rand::thread_rng();

    // Vector to store matrix sizes
    let mut sizes = Vec::new();

    // Add square matrices (e.g., 2x2, 8x8, etc.)
    for &size in &square_sizes {
        sizes.push((size, size, size)); // (n, m, p) for square matrices
    }

    // Add only valid rectangular matrices (e.g., (2x8 * 8x2), (8x16 * 16x8), etc.)
    for &(n, m) in &rectangular_sizes {
        sizes.push((n, m, n)); // (n x m) * (m x n)  → e.g., (2x8 * 8x2)
    }

    // Create a CSV writer
    let file = std::fs::File::create("benchmark_results.csv")?;
    let mut wtr = Writer::from_writer(file);

    // Write the header to the CSV file
    wtr.write_record(&[
        "Matrix Size (n)",
        "Matrix Size (m)",
        "Matrix Size (p)",
        "Memory Usage (MB)",
        "Average Memory Transfer Time (s)",
        "Average Computation Time (s)",
        "Average Execution Time (s)",
        "FLOP/s",
    ])?;

    // Perform benchmarking for each matrix size
    for &(n, m, p) in sizes.iter() {
        // Fill matrices with random f64 values between 0 and 100
        let a: Vec<f64> = (0..n * m)
            .map(|_| rng.gen_range(0.0..100.0)) // Random f64 between 0 and 100
            .collect();

        let b: Vec<f64> = (0..m * p)
            .map(|_| rng.gen_range(0.0..100.0)) // Random f64 between 0 and 100
            .collect();

        // Benchmark the GPU matrix multiplication
        benchmark_gpu(&a, &b, n, m, p, sample_size, &mut wtr)?;
    }

    // Flush the CSV writer to ensure all data is written to the file
    wtr.flush()?;

    Ok(())
}
