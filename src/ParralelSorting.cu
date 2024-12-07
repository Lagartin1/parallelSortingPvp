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
#define BLOCK_SIZE 256 // Tamaño del bloque para la GPU
using namespace std;

// Declaraciones de funciones
void generate_random_data(int* data, size_t n);
void merge(int* arr, int* left, int left_size, int* right, int right_size);
void parallel_merge_sort(int* arr, int n);
void cpu_parallel_sort(int* arr, size_t n);
void radixSortGPU(int* h_arr, int n);
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
        start_time = omp_get_wtime();
        radixSortGPU(data.data(), n);
        end_time = omp_get_wtime();
    }

    cout << n << " elementos ordenados en " << (end_time - start_time) << " segundos.\n";

    return 0;
}

// Implementación de funciones

// Genera datos aleatorios en un array de enteros
void generate_random_data(int* data, size_t n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 99999 );

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

// Kernel para contar las frecuencias de los dígitos
__global__ void countDigits(int* d_arr, int* d_count, int n, int exp) {
    __shared__ int local_count[10]; // Memoria compartida para los contadores

    // Inicializar los contadores locales
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = threadIdx.x;

    if (index < 10) local_count[index] = 0;
    __syncthreads();

    // Contar los dígitos en el lugar significativo actual
    if (tid < n) {
        int digit = (d_arr[tid] / exp) % 10;
        atomicAdd(&local_count[digit], 1);
    }

}

// Kernel para calcular los índices finales (scanning)
__global__ void computeOffsets(int* d_count, int* d_offset, int n) {
    int tid = threadIdx.x;

    if (tid == 0) {
        d_offset[0] = 0;
        for (int i = 1; i < 10; i++) {
            d_offset[i] = d_offset[i - 1] + d_count[i - 1];
        }
    }
}

// Kernel para reordenar los elementos en función del dígito actual
__global__ void reorderElements(int* d_arr, int* d_output, int* d_offset, int n, int exp) {
    __shared__ int local_offset[10];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < 10) {
        local_offset[threadIdx.x] = d_offset[threadIdx.x];
    }
    __syncthreads();

    if (tid < n) {
        int digit = (d_arr[tid] / exp) % 10;
        int position = atomicAdd(&local_offset[digit], 1);
        d_output[position] = d_arr[tid];
    }
}

// Función principal de Radix Sort en GPU
void radixSortGPU(int* h_arr, int n) {
    int* d_arr;
    int* d_output;
    int* d_count;
    int* d_offset;

    // Reservar memoria en el dispositivo
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));
    cudaMalloc((void**)&d_count, 10 * sizeof(int));
    cudaMalloc((void**)&d_offset, 10 * sizeof(int));

    // Copiar datos al dispositivo
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int max_val = *std::max_element(h_arr, h_arr + n);
    int exp = 1;

    // Número de bloques y threads
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    while (max_val / exp > 0) {
        // Inicializar los contadores
        cudaMemset(d_count, 0, 10 * sizeof(int));

        // Contar los dígitos
        countDigits<<<num_blocks, BLOCK_SIZE>>>(d_arr, d_count, n, exp);
        cudaDeviceSynchronize();

        // Calcular los offsets
        computeOffsets<<<1, 10>>>(d_count, d_offset, n);
        cudaDeviceSynchronize();

        // Reordenar los elementos
        reorderElements<<<num_blocks, BLOCK_SIZE>>>(d_arr, d_output, d_offset, n, exp);
        cudaDeviceSynchronize();

        // Copiar el resultado al arreglo original
        cudaMemcpy(d_arr, d_output, n * sizeof(int), cudaMemcpyDeviceToDevice);

        exp *= 10;
    }

    // Copiar el resultado final al host
    cudaMemcpy(h_arr, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria en el dispositivo
    cudaFree(d_arr);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaFree(d_offset);
}
