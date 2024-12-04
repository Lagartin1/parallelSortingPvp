NVCC = nvcc
NVCC_FLAGS = -Xcompiler -fopenmp -O3 -lineinfo
SRC_DIR = src
BIN_DIR = bin
SRC_FILE = $(wildcard $(SRC_DIR)/*.cu)
TARGET = $(BIN_DIR)/prog
# Asegurar que los directorios existen
$(shell mkdir -p $(BIN_DIR))

all: clean $(TARGET)

$(TARGET): $(SRC_FILE)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: clean all
