# Usa una imagen base de NVIDIA con soporte CUDA
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# Establece el directorio de trabajo
WORKDIR /workspace

# Instala herramientas esenciales
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia todo el proyecto al contenedor
COPY . .
