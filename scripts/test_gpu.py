import subprocess
import os
import re

# Rutas
bin_dir = "../bin"
results_dir = "../results"
program_path = os.path.join(bin_dir, "prog")
output_file = os.path.join(results_dir, "resultados_gpu.txt")

# Crear carpetas si no existen
os.makedirs(results_dir, exist_ok=True)

# Rango de tamaños de n (2^10 a 2^29)
n_values = [2**i for i in range(10, 30)]

# Expresión regular para extraer el tiempo
time_pattern = re.compile(r"(\d+\.?\d*e?-?\d*) segundos")

# Abre el archivo para escribir los resultados
with open(output_file, "w") as file:
    file.write("n,time\n")  # Escribe la cabecera del archivo

    # Itera sobre los tamaños de n
    for n in n_values:
        try:
            # Llama al programa con los argumentos (usando 1 hilo por defecto)
            result = subprocess.run(
                [program_path, str(n), "1", "1"],  # Modo CPU (0) y 1 hilo
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
                    file.write(f"{n},{elapsed_time:.6e}\n")
                    print(f"n={n}, time={elapsed_time:.6e}s")
                else:
                    print(f"No se encontró el tiempo en la salida para n={n}")
            else:
                print(f"Error ejecutando para n={n}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Timeout para n={n}")
        except Exception as e:
            print(f"Error inesperado para n={n}: {str(e)}")

print(f"Pruebas completadas. Resultados guardados en {output_file}.")
