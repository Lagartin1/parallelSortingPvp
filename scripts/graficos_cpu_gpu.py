import os
import numpy as np
import matplotlib.pyplot as plt

# Funciones auxiliares para calcular speedup y eficiencia paralela
def calculate_speedup(base_time, parallel_times):
    return base_time / parallel_times

def calculate_parallel_efficiency(speedup, num_blocks):
    return speedup / num_blocks

# Leer datos desde archivos en la carpeta "results"
def read_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [float(line.strip()) for line in lines]

# [Solo GPU] Speedup vs num-bloques
def plot_gpu_speedup(results_gpu, num_sms, output_dir):
    num_blocks = np.arange(1, num_sms * 5 + 1)
    base_time = results_gpu[0]
    speedups = calculate_speedup(base_time, results_gpu[:len(num_blocks)])

    plt.figure()
    plt.plot(num_blocks, speedups, marker='o', label="Speedup (GPU)")
    plt.xlabel("Número de bloques CUDA")
    plt.ylabel("Speedup")
    plt.title("[Solo GPU] Speedup vs Número de bloques")
    plt.legend()
    plt.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "speedup_vs_blocks.png"))
    plt.show()

# [Solo GPU] Eficiencia paralela vs num-bloques
def plot_gpu_efficiency(results_gpu, num_sms, output_dir):
    num_blocks = np.arange(1, num_sms * 5 + 1)
    base_time = results_gpu[0]
    speedups = calculate_speedup(base_time, results_gpu[:len(num_blocks)])
    efficiencies = calculate_parallel_efficiency(speedups, num_blocks)

    plt.figure()
    plt.plot(num_blocks, efficiencies, marker='o', label="Eficiencia paralela (GPU)")
    plt.xlabel("Número de bloques CUDA")
    plt.ylabel("Eficiencia paralela")
    plt.title("[Solo GPU] Eficiencia paralela vs Número de bloques")
    plt.legend()
    plt.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "efficiency_vs_blocks.png"))
    plt.show()

# Tiempo vs n (ambos, CPU y GPU)
def plot_time_vs_n(results_cpu, results_gpu, n_values, output_dir):
    plt.figure()
    plt.plot(n_values, results_cpu, marker='o', label="CPU")
    plt.plot(n_values, results_gpu, marker='x', label="GPU")
    plt.xlabel("Tamaño de problema (n)")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Tiempo vs Tamaño de problema (n)")
    plt.legend()
    plt.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "time_vs_n.png"))
    plt.show()

def main():
    # Leer datos de resultados
    resultados_cpu = read_results("results/resultados_cpu.txt")
    resultados_gpu = read_results("results/resultados_gpu.txt")

    # Definir valores de n y cantidad de SMs (estimado)
    n_values = np.logspace(3, 8, num=len(resultados_cpu), base=10, dtype=int)  # Desde 1000 hasta cientos de millones
    num_sms = 8  # Cambiar al número real de SMs de tu GPU

    # Directorio de salida para los gráficos
    output_dir = "graphics"

    # Graficar speedup y eficiencia para GPU
    plot_gpu_speedup(resultados_gpu, num_sms, output_dir)
    plot_gpu_efficiency(resultados_gpu, num_sms, output_dir)

    # Graficar tiempos para CPU y GPU
    plot_time_vs_n(resultados_cpu, resultados_gpu, n_values, output_dir)

if __name__ == "__main__":
    main()
