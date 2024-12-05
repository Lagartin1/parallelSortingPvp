#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Constantes para el Sort en GPU
#define BLOCK_SIZE 256
#define BUCKET_COUNT 8 // Número de buckets


using namespace std;
void generate_random_data(int* data, size_t n);
void merge(int* arr, int* left, int left_size, int* right, int right_size);
void parallel_merge_sort(int* arr, int n);
void cpu_parallel_sort(int* arr, size_t n);
__global__ void distribute_kernel(int* data, int* buckets, int* bucket_sizes, int n, int* pivots);
__global__ void local_sort_kernel(int* buckets, int* bucket_sizes, int bucket_idx, int bucket_capacity);
void gpu_sample_sort(int* data, size_t n);


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
        gpu_sample_sort(data.data(), n);
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
__global__ void distribute_kernel(int* data, int* buckets, int* bucket_sizes, int n, int* pivots) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int val = data[tid];
        int bucket = 0;

        // Determinar en qué bucket cae el valor
        while (bucket < BUCKET_COUNT - 1 && val > pivots[bucket]) {
            bucket++;
        }

        // Incrementar el tamaño del bucket y agregar el valor
        int pos = atomicAdd(&bucket_sizes[bucket], 1);
        buckets[bucket * n + pos] = val;
    }
}

__global__ void local_sort_kernel(int* buckets, int* bucket_sizes, int bucket_idx, int bucket_capacity) {
    extern __shared__ int shared_data[]; // Memoria compartida
    int tid = threadIdx.x;

    // Tamaño del bucket actual
    int size = bucket_sizes[bucket_idx];
    if (tid >= size) return;

    // Copiar los datos del bucket a la memoria compartida
    int* bucket_data = buckets + bucket_idx * bucket_capacity;
    shared_data[tid] = bucket_data[tid];
    __syncthreads();

    // Variables para el stack explícito
    int left_stack[32];
    int right_stack[32];
    int stack_size = 0;

    // Inicializar el stack con los límites del arreglo
    if (tid == 0) {
        left_stack[stack_size] = 0;
        right_stack[stack_size] = size - 1;
        stack_size++;
    }
    __syncthreads();

    // Procesar el stack
    while (stack_size > 0) {
        int left, right;

        if (tid == 0) {
            // Extraer límites del stack
            stack_size--;
            left = left_stack[stack_size];
            right = right_stack[stack_size];
        }
        __syncthreads();

        // Elegir el pivote como el elemento del medio
        int pivot = shared_data[(left + right) / 2];

        // Particionar el arreglo
        int i = left;
        int j = right;

        while (i <= j) {
            while (shared_data[i] < pivot) i++;
            while (shared_data[j] > pivot) j--;

            if (i <= j) {
                // Intercambiar
                int temp = shared_data[i];
                shared_data[i] = shared_data[j];
                shared_data[j] = temp;
                i++;
                j--;
            }
        }
        __syncthreads();

        // Añadir nuevos subarreglos al stack
        if (tid == 0) {
            if (left < j) {
                left_stack[stack_size] = left;
                right_stack[stack_size] = j;
                stack_size++;
            }
            if (i < right) {
                left_stack[stack_size] = i;
                right_stack[stack_size] = right;
                stack_size++;
            }
        }
        __syncthreads();
    }

    // Copiar los datos ordenados de vuelta al bucket
    bucket_data[tid] = shared_data[tid];
}


void gpu_sample_sort(int* data, size_t n) {
    int* d_data, *d_buckets, *d_bucket_sizes, *d_pivots;
    int* pivots = new int[BUCKET_COUNT - 1];
    int bucket_capacity = n;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sms = prop.multiProcessorCount;
    int num_blocks = num_sms * 5;
    // Generar pivotes (muestreo uniforme)
    for (int i = 0; i < BUCKET_COUNT - 1; i++) {
        pivots[i] = (i + 1) * (INT_MAX / BUCKET_COUNT);
    }

    // Reservar memoria en GPU
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_buckets, BUCKET_COUNT * bucket_capacity * sizeof(int));
    cudaMalloc(&d_bucket_sizes, BUCKET_COUNT * sizeof(int));
    cudaMalloc(&d_pivots, (BUCKET_COUNT - 1) * sizeof(int));

    // Copiar datos y pivotes al dispositivo
    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pivots, pivots, (BUCKET_COUNT - 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_bucket_sizes, 0, BUCKET_COUNT * sizeof(int));

    // Paso 1: Distribuir elementos en buckets
    distribute_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_buckets, d_bucket_sizes, n, d_pivots);

    // Paso 2: Ordenar localmente cada bucket
    for (int i = 0; i < BUCKET_COUNT; i++) {
        int shared_memory_size = bucket_capacity * sizeof(int);
        local_sort_kernel<<<1, bucket_capacity, shared_memory_size>>>(d_buckets, d_bucket_sizes, i, bucket_capacity);
        cudaDeviceSynchronize();
     }

    // Paso 3: Combinar buckets
    int offset = 0;
    for (int i = 0; i < BUCKET_COUNT; i++) {
        int size;
        cudaMemcpy(&size, &d_bucket_sizes[i], sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(data + offset, d_buckets + i * bucket_capacity, size * sizeof(int), cudaMemcpyDeviceToHost);
        offset += size;
    }

    // Liberar memoria
    delete[] pivots;
    cudaFree(d_data);
    cudaFree(d_buckets);
    cudaFree(d_bucket_sizes);
    cudaFree(d_pivots);
}