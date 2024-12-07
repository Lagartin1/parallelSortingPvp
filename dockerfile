# Usa una imagen base de NVIDIA con soporte CUDA
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Instala herramientas esenciales
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /workspace

# Copia todos los archivos de tu proyecto al contenedor
COPY . .

# Compila el programa (ajusta según tu compilación)
RUN nvcc -o prog main.cu -fopenmp

# Establece el comando por defecto al ejecutar el contenedor
CMD ["./prog", "100000", "1", "4"]
