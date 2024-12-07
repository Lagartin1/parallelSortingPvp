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

# Leer datos de std::sort en formato: n,tiempo
def read_std_results(file_path):
    data = np.loadtxt(file_path, delimiter=",", dtype=float, skiprows=1)  # Ignorar la primera fila (cabeceras)
    n_values = data[:, 0]
    times = data[:, 1]
    return n_values, times

# Calcular speedup
def calculate_speedup(base_times, test_times):
    return base_times / test_times

# Graficar Speedup vs n
def plot_speedup_vs_n(n_values, speedup_cpu, speedup_gpu, output_dir):
    # Graficar
    plt.figure()
    plt.plot(n_values, speedup_cpu, marker='o', label="Speedup (CPU)")
    plt.plot(n_values, speedup_gpu, marker='x', label="Speedup (GPU)")
    plt.xlabel("Tamaño de problema (n)")
    plt.ylabel("Speedup (relativo a std::sort)")
    plt.title("Speedup vs Tamaño de problema (n)")
    plt.legend()
    plt.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "speedup_vs_n.png"))
    plt.show()

def main():
    # Definir rutas relativas desde el directorio `scripts`
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio actual (`scripts`)
    results_dir = os.path.join(base_dir, "../results")     # Carpeta `results` (relativa)
    output_dir = os.path.join(base_dir, "../graphics")    # Carpeta `graphics` (relativa)

    # Leer datos de resultados
    n_values_cpu, threads_cpu, times_cpu = read_cpu_results(os.path.join(results_dir, "resultados_cpu.txt"))
    n_values_gpu, times_gpu = read_gpu_results(os.path.join(results_dir, "resultados_gpu.txt"))
    n_values_std, times_std = read_std_results(os.path.join(results_dir, "resultados_std.txt"))

    # Encontrar valores comunes de n
    common_n = np.intersect1d(np.intersect1d(n_values_cpu, n_values_gpu), n_values_std)

    # Filtrar los datos para que coincidan con common_n
    times_cpu = times_cpu[np.isin(n_values_cpu, common_n)]
    times_gpu = times_gpu[np.isin(n_values_gpu, common_n)]
    times_std = times_std[np.isin(n_values_std, common_n)]
    common_n = common_n[:min(len(times_cpu), len(times_gpu), len(times_std))]  # Ajustar tamaños

    # Ajustar los tiempos también
    times_cpu = times_cpu[:len(common_n)]
    times_gpu = times_gpu[:len(common_n)]
    times_std = times_std[:len(common_n)]

    # Graficar speedup vs n
    speedup_cpu = calculate_speedup(times_std, times_cpu)
    speedup_gpu = calculate_speedup(times_std, times_gpu)
    plot_speedup_vs_n(common_n, speedup_cpu, speedup_gpu, output_dir)

if __name__ == "__main__":
    main()