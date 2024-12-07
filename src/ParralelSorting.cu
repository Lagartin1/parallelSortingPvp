#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Constantes para el Sort en GPU
#define BLOCK_SIZE 512
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)
#define RADIX_MASK ((1 << RADIX_BITS) - 1)

using namespace std;
void generate_random_data(int* data, size_t n);
void merge(int* arr, int* left, int left_size, int* right, int right_size);
void parallel_merge_sort(int* arr, int n);
void cpu_parallel_sort(int* arr, size_t n);
__global__ void count_kernel(int* input, int* count, int n, int shift);
__global__ void scatter_kernel(int* input, int* output, int* global_offset, int n, int shift);

void gpu_radix_sort(int* data, int n);


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
    vector<int> verify_data(n); // Array para verificación
    generate_random_data(data.data(), n); // Generar datos aleatorios
    verify_data = data;

    double start_time, end_time;

    if (mode == 0) { // Modo CPU
        omp_set_num_threads(num_threads);
        start_time = omp_get_wtime();
        cpu_parallel_sort(data.data(), n);
        end_time = omp_get_wtime();
    } 
    else if (mode == 1) { // Modo GPU
        start_time = omp_get_wtime();
        gpu_radix_sort(data.data(), n);
        end_time = omp_get_wtime();
    }

    sort(verify_data.begin(), verify_data.end());
    bool is_correct = equal(data.begin(), data.end(), verify_data.begin());

    if (!is_correct) {
        cerr << "Error: El ordenamiento no es correcto!\n";
        return 1;
    }
    cout << "El ordenamiento es correcto!\n";
    cout << n << " elementos ordenados en " << end_time - start_time << " segundos.\n";

    return 0;
}

// Implementación de funciones

// Genera datos aleatorios en un array de enteros
void generate_random_data(int* data, size_t n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, INT_MAX);

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


// Kernel para contar las ocurrencias de cada dígito en Radix Sort
__global__ void count_kernel(int* input, int* count, int n, int shift) {
    __shared__ int local_count[RADIX_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < RADIX_SIZE) {
        local_count[tid] = 0;
    }
    __syncthreads();

    if (gid < n) {
        int digit = (input[gid] >> shift) & RADIX_MASK;
        atomicAdd(&local_count[digit], 1);
    }
    __syncthreads();

    if (tid < RADIX_SIZE) {
        atomicAdd(&count[tid], local_count[tid]);
    }
}

// Kernel para distribuir elementos según sus dígitos
__global__ void scatter_kernel(int* input, int* output, int* global_offset, int n, int shift) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        int digit = (input[gid] >> shift) & RADIX_MASK;
        int pos = atomicAdd(&global_offset[digit], 1);
        output[pos] = input[gid];
    }
}

// Implementación de Radix Sort en GPU
void gpu_radix_sort(int* data, int n) {
    int* d_input, *d_output, *d_count, *d_offsets;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_count, RADIX_SIZE * sizeof(int));
    cudaMalloc(&d_offsets, RADIX_SIZE * sizeof(int));

    cudaMemcpy(d_input, data, n * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        cudaMemset(d_count, 0, RADIX_SIZE * sizeof(int));
        count_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_count, n, shift);
        cudaDeviceSynchronize();

        int count[RADIX_SIZE];
        cudaMemcpy(count, d_count, RADIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

        int total = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            int temp = count[i];
            count[i] = total;
            total += temp;
        }

        cudaMemcpy(d_offsets, count, RADIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        scatter_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, d_offsets, n, shift);
        cudaDeviceSynchronize();

        swap(d_input, d_output);
    }

    cudaMemcpy(data, d_input, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaFree(d_offsets);
}