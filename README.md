# Parallel Sorting PVP

Este proyecto implementa y compara dos algoritmos de ordenamiento paralelos investigados como los mas rapidos:
1. **Merge Sort paralelo** en CPU utilizando OpenMP.
2. **Radix Sort paralelo** en GPU utilizando CUDA.

El objetivo es medir el rendimiento de ambos algoritmos en diferentes configuraciones y tamaños de entrada.


# Requisitos

## Hardware
- GPU compatible con CUDA.
- CPU con soporte para múltiples núcleos para OpenMP.

## Software
- **CUDA Toolkit** (versión 12.7 o superior).
- Compilador compatible con CUDA (`nvcc`).
- Compilador C++ con soporte para OpenMP (por ejemplo, `g++`).
- Sistema operativo basado en Linux.


# Instalación recursos

## instalar CUDA Toolkit (Ubuntu)
Para instalar CUDA Toolkit en Ubuntu, se puede utilizar el siguiente comando:
```bash
sudo apt-get install nvidia-cuda-toolkit
```
para otros sistmas y mas informacion en: https://developer.nvidia.com/cuda-dowloads

## para windows subsystem for linux 2 (wsl2):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```
mas infomacion en: https://developer.nvidia.com/cuda/wsl
y en: https://developer.nvidia.com/cuda-downloads


# Clonar el repositorio
```bash
git clone https://github.com/tuusuario/parallelSortingPvp.git
cd parallelSortingPvp
```

# Compilar el código
```bash
make
```
# Ejecutar el programa
El programa se ejecuta con en la carpeta `bin`,por lo tanto ejecute los siguinetes comandos:
```bash
cd bin
./prog <n> <modo> <nt>
```
Donde:
- `n` es el tamaño del arreglo a ordenar.
- `modo` es el tipo de algoritmo a utilizar (0 para CPU, 1 para GPU).
- `nt` es el número de threads a utilizar en el caso de CPU.
## Ejemplo de ejecucion
```bash
./prog $((2**24)) 0 8
```
## Resultados
```bash
➜ $ ./prog $((2**24)) 0 8   
El ordenamiento es correcto!
16777216 elementos ordenados en 0.838904 segundos.
```
# Ejecucion script
## scripts de pruebas:
debe estar en el directorio `/scripts` ejecute el siguiente comando:
```bash
python3 test_cpu.py
```
los resultados se guardaran en la carpeta `/results`

## scripts de graficos solo CPU:
debe estar en el directorio `/scripts` ejecute el siguiente comando:
```bash
python3 graficos_Cpu.py
```
los graficos se guardaran en la carpeta `/graphics