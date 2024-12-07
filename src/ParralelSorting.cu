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
void checkGPUMemory();
void printGPUProperties();
void gpu_radix_sort(int* data, size_t n);

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
        // Reservar memoria host para el sorting en GPU
        int* h_data = (int*)malloc(n * sizeof(int));
        if (!h_data) {
            cerr << "Error: no se pudo asignar memoria host para h_data.\n";
            return 1;
        }
        // Copiar data a h_data
        std::copy(data.begin(), data.end(), h_data);

        start_time = omp_get_wtime();
        // Llamar a gpu_radix_sort que asume que h_data es host
        gpu_radix_sort(h_data, n);
        end_time = omp_get_wtime();

        // Copiar resultado a data
        std::copy(h_data, h_data + n, data.begin());
        free(h_data);
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

// Kernel para generar el predicate basado en un bit dado
__global__ void predicate_kernel(const int* __restrict__ d_input, int* __restrict__ d_predicate, int bit, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = d_input[idx];
        int bit_val = (val >> bit) & 1;
        d_predicate[idx] = bit_val; // 1 si el bit es 1, 0 si es 0
    }
}

// Kernel para aplicar el resultado del reorder usando predicate y su complemento
__global__ void reorder_kernel(const int* __restrict__ d_input,
                               int* __restrict__ d_output,
                               const int* __restrict__ d_predicate,
                               const int* __restrict__ d_predicate_scan,
                               const int* __restrict__ d_ters_predicate_scan,
                               int num_ones,
                               size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int p = d_predicate[idx];
        if (p == 1) {
            // Índice para los que tienen el bit en 1
            int new_index = d_predicate_scan[idx];
            d_output[new_index] = d_input[idx];
        } else {
            // Índice para los que tienen el bit en 0
            int new_index = d_ters_predicate_scan[idx] + num_ones;
            d_output[new_index] = d_input[idx];
        }
    }
}

// Función principal Radix Sort (bit a bit usando predicate)
void gpu_radix_sort(int* data, size_t n) {
    if (n == 0) return;

    // Memoria en GPU
    int *d_input, *d_output, *d_predicate;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, n*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, n*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_predicate, n*sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, data, n*sizeof(int), cudaMemcpyHostToDevice));

    // Vectores de thrust para scans
    thrust::device_vector<int> d_predicate_scan(n);
    thrust::device_vector<int> d_ters_predicate(n);
    thrust::device_vector<int> d_ters_predicate_scan(n);

    int passes = BITS; // para 32 bits

    for (int bit = 0; bit < passes; bit += RADIX_BITS) {
        // Calcular predicate
        {
            int num_blocks = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            predicate_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_predicate, bit, n);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        // Copiar predicate a vector thrust
        thrust::device_ptr<int> pred_ptr(d_predicate);
        thrust::copy(pred_ptr, pred_ptr + n, d_predicate_scan.begin());

        // Calcular prefix sum del predicate (esto da las posiciones de los que tienen bit=1)
        thrust::exclusive_scan(d_predicate_scan.begin(), d_predicate_scan.end(), d_predicate_scan.begin());

        // Contar cuantos 1 hay: el último valor en la prefix sum + el valor del último elemento
        // Para saber el num_ones: sum = (última posición en d_predicate_scan) + ultimo valor d_predicate
        int last_pred_val = d_predicate_scan[n-1] + ( (int)thrust::device_pointer_cast(d_predicate)[n-1] );
        int num_ones;
        CHECK_CUDA_ERROR(cudaMemcpy(&num_ones, thrust::raw_pointer_cast(&d_predicate_scan[n-1]), sizeof(int), cudaMemcpyDeviceToHost));
        // sumarle el valor real del último predicate para contar correctamente
        int last_bit;
        CHECK_CUDA_ERROR(cudaMemcpy(&last_bit, d_predicate + (n-1), sizeof(int), cudaMemcpyDeviceToHost));
        num_ones += last_bit;

        // ters_predict = !predicate
        thrust::transform(pred_ptr, pred_ptr+n, d_ters_predicate.begin(), thrust::logical_not<int>());

        // Scan de ters_predict
        thrust::exclusive_scan(d_ters_predicate.begin(), d_ters_predicate.end(), d_ters_predicate_scan.begin());

        // Ahora hacer reorder usando reorder_kernel
        {
            int num_blocks = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            reorder_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input,
                                                       d_output,
                                                       d_predicate,
                                                       thrust::raw_pointer_cast(d_predicate_scan.data()),
                                                       thrust::raw_pointer_cast(d_ters_predicate_scan.data()),
                                                       num_ones, n);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        // Intercambiar punteros
        int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    // Copiar resultado final
    CHECK_CUDA_ERROR(cudaMemcpy(data, d_input, n*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_predicate));
}
