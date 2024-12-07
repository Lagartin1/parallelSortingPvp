#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

// Parámetros del Radix Sort
#define RADIX_BITS 1  // para emular el bit-a-bit (LSD) del ejemplo
#define BITS 32
#define BLOCK_SIZE 256
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

using namespace std;

// Declaraciones de funciones
void generate_random_data(int* data, size_t n);
void merge(int* arr, int* left, int left_size, int* right, int right_size);
void parallel_merge_sort(int* arr, int n);
void cpu_parallel_sort(int* arr, size_t n);
__global__ void maskKernel(int *d_input, int *d_mask, int bit, int n);
__global__ void computeScanKernel(int *d_mask, int *d_scan, int n);
__global__ void KernelScatter(int *d_input, int *d_output, int *d_mask, int *d_scan, int n, int totalZeros);

void gpu_radix_sort(int *arr, int n, int gridSize);

// Función principal
int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "Uso: ./prog <n> <modo> <nt>\n";
        cout << "  n: tamaño del array\n";
        cout << "  modo: 0 para CPU, 1 para GPU \n";
        cout << "  nt: número de threads (CPU)\n";
        return 1;
    }

    size_t n = stoull(argv[1]); // Tamaño del array
    int mode = stoi(argv[2]);  // Modo: CPU o GPU
    int num_threads = stoi(argv[3]); // Número de threads para CPU
    if (mode == 0 && num_threads <= 0) {
        cerr << "Error: número de threads inválido.\n";
        return 1;
    }
    if (mode != 0 && mode != 1) {
        cerr << "Error: modo inválido. Use 0 para CPU o 1 para GPU.\n";
        return 1;
    }

    vector<int> data(n); // Array de datos
    generate_random_data(data.data(), n); // Generar datos aleatorios
    vector<int> verify_data = data;       // Copia para verificación

    double start_time, end_time;

    if (mode == 0) { // Modo CPU
        omp_set_num_threads(num_threads);
        start_time = omp_get_wtime();
        cpu_parallel_sort(data.data(), n);
        end_time = omp_get_wtime();
    } 
    else if (mode == 1) { // Modo GPU
        int blocks = BLOCK_SIZE;
        start_time = omp_get_wtime();
        gpu_radix_sort(data.data(), n, blocks);
        end_time = omp_get_wtime();
    }

    // Verificación
    sort(verify_data.begin(), verify_data.end());
    bool is_correct = equal(data.begin(), data.end(), verify_data.begin());

    if (!is_correct) {
        cerr << "Error: El ordenamiento no es correcto!\n";
        return 1;
    }
    cout << "El ordenamiento es correcto!\n";
    cout << n << " elementos ordenados en " << (end_time - start_time) << " segundos.\n";

    return 0;
}

// Implementación de funciones

// Genera datos aleatorios en un array de enteros
void generate_random_data(int* data, size_t n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, (1<<16)-1);

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
}

// Mezcla dos subarrays ordenados en un array resultante
void merge(int* arr, int* left, int left_size, int* right, int right_size) {
    int i = 0, j = 0, k = 0;
    vector<int> temp(left_size + right_size);

    while (i < left_size && j < right_size) {
        if (left[i] <= right[j]) {
            temp[k++] = left[i++];
        } else {
            temp[k++] = right[j++];
        }
    }

    while (i < left_size) temp[k++] = left[i++];
    while (j < right_size) temp[k++] = right[j++];


    copy(temp.begin(), temp.end(), arr);
}

// Implementa Merge Sort en paralelo usando OpenMP
void parallel_merge_sort(int* arr, int n) {
    if (n <= 1) return;

    int mid = n / 2;

    #pragma omp task shared(arr) if(n > 100000)
    parallel_merge_sort(arr, mid);

    #pragma omp task shared(arr) if(n > 100000)
    parallel_merge_sort(arr + mid, n - mid);

    #pragma omp taskwait
    merge(arr, arr, mid, arr + mid, n - mid);
}

// Función principal para Merge Sort paralelo
void cpu_parallel_sort(int* arr, size_t n) {
    #pragma omp parallel
    {
        #pragma omp single
        parallel_merge_sort(arr, n);
    }
}

// kernel para calcular la mask de bits en Radix Sort
__global__ void maskKernel(int *d_input, int *d_mask, int bit, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_mask[idx] = (d_input[idx] >> bit) & 1;
    }
}

// kernel para calcular el scan exclusivo en Radix Sort
__global__ void computeScanKernel(int *d_mask, int *d_scan, int n) {
    extern __shared__ int temp[];
    int idx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + idx;

    if (i < n) {
        temp[idx] = d_mask[i];
    } else {
        temp[idx] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = idx >= offset ? temp[idx - offset] : 0;
        __syncthreads();
        temp[idx] += val;
        __syncthreads();
    }

    if (i < n) {
        d_scan[i] = temp[idx];
    }
}

// kernel para hacer scatter en Radix Sort
__global__ void KernelScatter(int *d_input, int *d_output, int *d_mask, int *d_scan, int n, int totalZeros) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int pos;
        if (d_mask[idx] == 0) {
            pos = d_scan[idx];
        } else {
            pos = totalZeros + idx - d_scan[idx];
        }
        d_output[pos] = d_input[idx];
    }
}

// implementación de Radix Sort en GPU
void gpu_radix_sort(int *arr, int n, int gridSize) {
    int *d_input, *d_output, *d_mask, *d_scan;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_mask, n * sizeof(int));
    cudaMalloc(&d_scan, n * sizeof(int));

    cudaMemcpy(d_input, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);

    for (int bit = 0; bit < 32; ++bit) {
        maskKernel<<<gridSize, blockSize>>>(d_input, d_mask, bit, n);
        cudaDeviceSynchronize();

        computeScanKernel<<<gridSize, blockSize, blockSize.x * sizeof(int)>>>(d_mask, d_scan, n);
        cudaDeviceSynchronize();

        int totalZeros;
        cudaMemcpy(&totalZeros, &d_scan[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
        totalZeros += 1 - ((arr[n - 1] >> bit) & 1);

        KernelScatter<<<gridSize, blockSize>>>(d_input, d_output, d_mask, d_scan, n, totalZeros);
        cudaDeviceSynchronize();

        swap(d_input, d_output);
    }

    cudaMemcpy(arr, d_input, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaFree(d_scan);
}
