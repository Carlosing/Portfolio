extern "C" __global__ void matrix_mul(const float *A, const float *B, float *C, int rows, int cols, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < size) {
        float sum = 0.0;
        
        // Imprime qué hilo está ejecutándose y el estado inicial de sum
        
        
        for (int i = 0; i < cols; i++) {
            float a_val = A[row * cols + i];
            float b_val = B[i * size + col];
            sum += a_val * b_val;

            
        }
        
        C[row * size + col] = sum;

        
        
    }
}
