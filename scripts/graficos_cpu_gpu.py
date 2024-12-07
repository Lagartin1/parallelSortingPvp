import os
import numpy as np
import matplotlib.pyplot as plt

# Leer datos de CPU en formato: n,threads,tiempo
def read_cpu_results(file_path):
    data = np.loadtxt(file_path, delimiter=",", dtype=float, skiprows=1)  # Ignorar la primera fila (cabeceras)
    n_values = data[:, 0]
    threads = data[:, 1]
    times = data[:, 2]
    return n_values, threads, times

# Leer datos de GPU en formato: n,tiempo
def read_gpu_results(file_path):
    data = np.loadtxt(file_path, delimiter=",", dtype=float, skiprows=1)  # Ignorar la primera fila (cabeceras)
    n_values = data[:, 0]
    times = data[:, 1]
    return n_values, times

# Funciones auxiliares para calcular speedup y eficiencia paralela
def calculate_speedup(base_time, parallel_times):
    return base_time / parallel_times

def calculate_parallel_efficiency(speedup, num_blocks):
    return speedup / num_blocks

# [Solo GPU] Speedup vs num-bloques
def plot_gpu_speedup(n_values_gpu, times_gpu, num_sms, output_dir):
    num_blocks = np.arange(1, len(times_gpu) + 1)  # Ajustar num_blocks a la cantidad de datos disponibles
    base_time = times_gpu[0]
    speedups = calculate_speedup(base_time, times_gpu[:len(num_blocks)])

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
def plot_gpu_efficiency(n_values_gpu, times_gpu, num_sms, output_dir):
    num_blocks = np.arange(1, len(times_gpu) + 1)  # Ajustar num_blocks a la cantidad de datos disponibles
    base_time = times_gpu[0]
    speedups = calculate_speedup(base_time, times_gpu[:len(num_blocks)])
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
def plot_time_vs_n(n_values_cpu, times_cpu, n_values_gpu, times_gpu, output_dir):
    plt.figure()
    plt.plot(n_values_cpu, times_cpu, marker='o', label="CPU")
    plt.plot(n_values_gpu, times_gpu, marker='x', label="GPU")
    plt.xlabel("Tamaño de problema (n)")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Tiempo vs Tamaño de problema (n)")
    plt.legend()
    plt.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "time_vs_n.png"))
    plt.show()

def main():
    # Definir rutas relativas desde el directorio `scripts`
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio actual (`scripts`)
    results_dir = os.path.join(base_dir, "../results")     # Carpeta `results` (relativa)
    output_dir = os.path.join(base_dir, "../graphics")    # Carpeta `graphics` (relativa)

    # Leer datos de resultados
    n_values_cpu, threads_cpu, times_cpu = read_cpu_results(os.path.join(results_dir, "resultados_cpu.txt"))
    n_values_gpu, times_gpu = read_gpu_results(os.path.join(results_dir, "resultados_gpu.txt"))

    # Definir cantidad de SMs (estimado)
    num_sms = 8  # Cambiar al número real de SMs de tu GPU

    # Graficar speedup y eficiencia para GPU
    plot_gpu_speedup(n_values_gpu, times_gpu, num_sms, output_dir)
    plot_gpu_efficiency(n_values_gpu, times_gpu, num_sms, output_dir)

    # Graficar tiempos para CPU y GPU
    plot_time_vs_n(n_values_cpu, times_cpu, n_values_gpu, times_gpu, output_dir)

if __name__ == "__main__":
    main()
