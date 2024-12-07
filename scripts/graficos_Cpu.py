import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os

# Rutas
results_dir = "../results"
graphics_dir = "../graphics"
input_file = os.path.join(results_dir, "resultados_cpu.txt")
speedup_plot = os.path.join(graphics_dir, "speedup_vs_threads.png")
efficiency_plot = os.path.join(graphics_dir, "efficiency_vs_threads.png")

# Crear carpetas si no existen
os.makedirs(results_dir, exist_ok=True)
os.makedirs(graphics_dir, exist_ok=True)

# Cargar los datos del archivo
data = pd.read_csv(input_file)

# Seleccionar un valor de n suficientemente alto
n_high = max(data["n"])

# Filtrar los datos para ese n
data_high_n = data[data["n"] == n_high]

# Detectar el número máximo de threads en los datos
max_threads = max(data_high_n["threads"])
num_cores = multiprocessing.cpu_count()  # Detectar núcleos disponibles

# Calcular el tiempo base (1 thread)
base_time = data_high_n[data_high_n["threads"] == 1]["time"].iloc[0]

# Calcular Speedup y Eficiencia Paralela
data_high_n["speedup"] = base_time / data_high_n["time"]
data_high_n["efficiency"] = data_high_n["speedup"] / data_high_n["threads"]

# Crear el gráfico de Speedup
plt.figure()
plt.plot(
    data_high_n["threads"], data_high_n["speedup"], marker="o", label="Speedup"
)
plt.axhline(y=num_cores, color="r", linestyle="--", label="Ideal Speedup")
plt.title(f"Speedup vs Número de Threads")
plt.xlabel("Número de Threads")
plt.ylabel("Speedup")
plt.legend()
plt.grid()
plt.savefig(speedup_plot)
plt.show()

# Crear el gráfico de Eficiencia Paralela
plt.figure()
plt.plot(
    data_high_n["threads"], data_high_n["efficiency"], marker="o", label="Eficiencia Paralela"
)
plt.axhline(y=1.0, color="r", linestyle="--", label="Eficiencia Ideal")
plt.title(f"Eficiencia Paralela vs Número de Threads")
plt.xlabel("Número de Threads")
plt.ylabel("Eficiencia Paralela")
plt.legend()
plt.grid()
plt.savefig(efficiency_plot)
plt.show()

print(f"Gráficos guardados en:\n{speedup_plot}\n{efficiency_plot}")
