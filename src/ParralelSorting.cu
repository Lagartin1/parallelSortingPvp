#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constantes para el Sort en GPU
#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)
#define RADIX_MASK ((1 << RADIX_BITS) - 1)
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
void checkGPUMemory();
void printGPUProperties();
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

// Función para verificar recursos de la GPU
void checkGPUMemory() {
    size_t free_mem, total_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    
    cout << "GPU Memory Info:" << endl;
    cout << "Total Memory: " << total_mem / (1024 * 1024) << " MB" << endl;
    cout << "Free Memory: " << free_mem / (1024 * 1024) << " MB" << endl;
}

// Función para obtener propiedades de la GPU
void printGPUProperties() {
    cudaDeviceProp prop;
    int deviceCount;
    
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    for (int device = 0; device < deviceCount; ++device) {
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
        
        cout << "GPU Device " << device << " Properties:" << endl;
        cout << "  Name: " << prop.name << endl;
        cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  Multiprocessor Count: " << prop.multiProcessorCount << endl;
        cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
        cout << "  Max Block Dimensions: " 
             << prop.maxThreadsDim[0] << " x " 
             << prop.maxThreadsDim[1] << " x " 
             << prop.maxThreadsDim[2] << endl;
        cout << "  Max Grid Dimensions: " 
             << prop.maxGridSize[0] << " x " 
             << prop.maxGridSize[1] << " x " 
             << prop.maxGridSize[2] << endl;
    }
}

// Kernel de conteo para Radix Sort
__global__ void count_kernel(int* input, int* count, size_t n, int shift) {
    __shared__ int local_count[RADIX_SIZE];

    int tid = threadIdx.x;
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    // Inicializar contadores locales
    if (tid < RADIX_SIZE) {
        local_count[tid] = 0;
    }
    __syncthreads();

    // Contar en bloques más grandes
    for (size_t i = gid; i < n; i += gridDim.x * blockDim.x) {
        int digit = (input[i] >> shift) & RADIX_MASK;
        atomicAdd(&local_count[digit], 1);
    }
    __syncthreads();

    // Actualizar contadores globales
    if (tid < RADIX_SIZE) {
        atomicAdd(&count[tid], local_count[tid]);
    }
}

// Kernel de dispersión para Radix Sort
__global__ void scatter_kernel(int* input, int* output, int* global_offset, size_t n, int shift) {
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    // Procesar en bloques más grandes
    for (size_t i = gid; i < n; i += gridDim.x * blockDim.x) {
        int digit = (input[i] >> shift) & RADIX_MASK;
        size_t pos = atomicAdd((unsigned int*)&global_offset[digit], 1);
        output[pos] = input[i];
    }
}

// Implementación de Radix Sort en GPU
void gpu_radix_sort(int* data, int n) {
    // Verificar recursos de GPU
    checkGPUMemory();

    // Calcular memoria necesaria
    size_t input_memory = (size_t)n * sizeof(int);
    size_t radix_memory = RADIX_SIZE * sizeof(int);

    // Punteros de dispositivo
    int *d_input = nullptr, *d_output = nullptr, *d_count = nullptr, *d_offsets = nullptr;

    try {
        // Asignación de memoria con verificación de errores
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_memory));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, input_memory));
        CHECK_CUDA_ERROR(cudaMalloc(&d_count, radix_memory));
        CHECK_CUDA_ERROR(cudaMalloc(&d_offsets, radix_memory));

        // Copiar datos de entrada
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, data, input_memory, cudaMemcpyHostToDevice));

        // Configuración de bloques
        int num_blocks = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535 * 4);

        // Radix Sort principal
        for (int shift = 0; shift < 32; shift += RADIX_BITS) {
            // Reiniciar contadores
            CHECK_CUDA_ERROR(cudaMemset(d_count, 0, radix_memory));
            
            // Kernel de conteo
            count_kernel<<<num_blocks, BLOCK_SIZE>>>((int*)d_input, d_count, n, shift);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Calcular prefijos
            int count[RADIX_SIZE] = {0};
            CHECK_CUDA_ERROR(cudaMemcpy(count, d_count, radix_memory, cudaMemcpyDeviceToHost));

            int total = 0;
            for (int i = 0; i < RADIX_SIZE; i++) {
                int temp = count[i];
                count[i] = total;
                total += temp;
            }

            // Copiar contadores de vuelta al dispositivo
            CHECK_CUDA_ERROR(cudaMemcpy(d_offsets, count, radix_memory, cudaMemcpyHostToDevice));

            // Kernel de dispersión
            scatter_kernel<<<num_blocks, BLOCK_SIZE>>>((int*)d_input, d_output, d_offsets, n, shift);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Intercambiar punteros
            swap(d_input, d_output);
        }

        // Copiar resultado de vuelta al host
        CHECK_CUDA_ERROR(cudaMemcpy(data, d_input, input_memory, cudaMemcpyDeviceToHost));
    }
    catch (const std::exception& e) {
        cerr << "Error during GPU sorting: " << e.what() << endl;
    }

    // Liberar memoria
    if (d_input) CHECK_CUDA_ERROR(cudaFree(d_input));
    if (d_output) CHECK_CUDA_ERROR(cudaFree(d_output));
    if (d_count) CHECK_CUDA_ERROR(cudaFree(d_count));
    if (d_offsets) CHECK_CUDA_ERROR(cudaFree(d_offsets));
}