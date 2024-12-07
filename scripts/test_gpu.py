import subprocess
import os
import re

# Rutas
bin_dir = "../bin"
results_dir = "../results"
program_path = os.path.join(bin_dir, "prog")
output_file = os.path.join(results_dir, "resultados_cpu.txt")

# Crear carpetas si no existen
os.makedirs(results_dir, exist_ok=True)

# Rango de tamaños de n (2^10 a 2^29)
n_values = [2**i for i in range(10, 30)]
# Número de threads (1 a 8)
thread_values = range(1, 9)

# Expresión regular para extraer el tiempo
time_pattern = re.compile(r"(\d+\.?\d*e?-?\d*) segundos")

# Abre el archivo para escribir los resultados
with open(output_file, "w") as file:
    file.write("n,threads,time\n")  # Escribe la cabecera del archivo

    # Itera sobre los tamaños de n y los threads
    for n in n_values:
        for threads in thread_values:
            try:
                # Llama al programa con los argumentos
                result = subprocess.run(
                    [program_path, str(n), "1", str(threads)],  # Modo CPU (0)
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600  # Tiempo límite por ejecución en segundos
                )

                # Verifica si el programa se ejecutó correctamente
                if result.returncode == 0:
                    # Buscar el tiempo en la salida
                    match = time_pattern.search(result.stdout)
                    if match:
                        elapsed_time = float(match.group(1))  # Extraer el tiempo como float
                        # Escribe los resultados
                        file.write(f"{n},{threads},{elapsed_time:.6e}\n")
                        print(f"n={n}, threads={threads}, time={elapsed_time:.6e}s")
                    else:
                        print(f"No se encontró el tiempo en la salida para n={n}, threads={threads}")
                else:
                    print(f"Error ejecutando para n={n}, threads={threads}: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"Timeout para n={n}, threads={threads}")
            except Exception as e:
                print(f"Error inesperado para n={n}, threads={threads}: {str(e)}")

print(f"Pruebas completadas. Resultados guardados en {output_file}.")
