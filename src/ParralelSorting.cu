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
#define MAX_BLOCK_SZ 128
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
void gpu_radix_sort(int* data, size_t n, int blocks);
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
        gpu_radix_sort(data.data(), n, MAX_BLOCK_SZ);
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

// GPU Radix Sort Kernels
__global__ void gpu_radix_sort_local(unsigned int* d_out_sorted, unsigned int* d_prefix_sums, unsigned int* d_block_sums,
                                     unsigned int input_shift_width, unsigned int* d_in, unsigned int d_in_len, unsigned int max_elems_per_block) {
    extern __shared__ unsigned int shmem[];
    unsigned int* s_data = shmem;
    unsigned int* s_mask_out = &s_data[max_elems_per_block];
    unsigned int* s_merged_scan_mask_out = &s_mask_out[max_elems_per_block + 1];
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
    if (cpy_idx < d_in_len)
        s_data[thid] = d_in[cpy_idx];
    else
        s_data[thid] = 0;

    __syncthreads();

    unsigned int t_data = s_data[thid];
    unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;

    for (unsigned int i = 0; i < 4; ++i) {
        s_mask_out[thid] = 0;
        if (thid == 0) s_mask_out[max_elems_per_block] = 0;

        __syncthreads();

        if (cpy_idx < d_in_len) {
            bool val_equals_i = t_2bit_extract == i;
            s_mask_out[thid] = val_equals_i;
        }

        __syncthreads();

        for (unsigned int d = 0; d < log2f(max_elems_per_block); ++d) {
            int partner = thid - (1 << d);
            unsigned int sum = (partner >= 0) ? s_mask_out[thid] + s_mask_out[partner] : s_mask_out[thid];
            __syncthreads();
            s_mask_out[thid] = sum;
            __syncthreads();
        }

        if (thid == 0) {
            s_mask_out[0] = 0;
            unsigned int total_sum = s_mask_out[max_elems_per_block];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }

        __syncthreads();

        if (cpy_idx < d_in_len) {
            unsigned int t_prefix_sum = s_mask_out[thid];
            unsigned int new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];
            __syncthreads();
            s_data[new_pos] = t_data;
        }

        __syncthreads();
    }

    if (cpy_idx < d_in_len) {
        d_out_sorted[cpy_idx] = s_data[thid];
    }
}

__global__ void gpu_glbl_shuffle(unsigned int* d_out, unsigned int* d_in, unsigned int* d_scan_block_sums,
                                 unsigned int* d_prefix_sums, unsigned int input_shift_width, unsigned int d_in_len, unsigned int max_elems_per_block) {
    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len) {
        unsigned int t_data = d_in[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];
        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x] + t_prefix_sum;
        __syncthreads();
        d_out[data_glbl_pos] = t_data;
    }
}

void gpu_radix_sort(int* data, size_t n, int blocks) {
    unsigned int block_sz = MAX_BLOCK_SZ;
    unsigned int max_elems_per_block = block_sz;
    unsigned int grid_sz = (n + max_elems_per_block - 1) / max_elems_per_block;

    unsigned int* d_in;
    unsigned int* d_out;
    cudaMalloc(&d_in, sizeof(unsigned int) * n);
    cudaMalloc(&d_out, sizeof(unsigned int) * n);
    cudaMemcpy(d_in, data, sizeof(unsigned int) * n, cudaMemcpyHostToDevice);

    unsigned int* d_prefix_sums;
    unsigned int d_prefix_sums_len = n;
    cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * d_prefix_sums_len);
    cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_prefix_sums_len);

    unsigned int* d_block_sums;
    unsigned int d_block_sums_len = 4 * grid_sz;
    cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int* d_scan_block_sums;
    cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int shmem_sz = (max_elems_per_block * 5) * sizeof(unsigned int);

    for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 2) {
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_out, d_prefix_sums, d_block_sums, shift_width, d_in, n, max_elems_per_block);
        cudaDeviceSynchronize();
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in, d_out, d_scan_block_sums, d_prefix_sums, shift_width, n, max_elems_per_block);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data, d_out, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_scan_block_sums);
    cudaFree(d_block_sums);
    cudaFree(d_prefix_sums);
    cudaFree(d_in);
    cudaFree(d_out);
}
