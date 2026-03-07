NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# RTX 5090 = SM 12.0
CUDA_ARCH = -gencode arch=compute_120,code=sm_120

NVCC_FLAGS = $(CUDA_ARCH) -O3 -std=c++17 --use_fast_math -Xcompiler -Wall -Ithird_party
LDFLAGS = -lcudart -lpthread

BUILD_DIR = build
SRC_DIR = src

# Source files (shared between CLI and server)
CU_SOURCES = \
    $(SRC_DIR)/sampling.cu \
    $(SRC_DIR)/kernels/rmsnorm.cu \
    $(SRC_DIR)/kernels/embedding.cu \
    $(SRC_DIR)/kernels/rope.cu \
    $(SRC_DIR)/kernels/attention.cu \
    $(SRC_DIR)/kernels/ffn.cu \
    $(SRC_DIR)/kernels/mamba.cu

CPP_SOURCES = \
    $(SRC_DIR)/gguf_loader.cpp \
    $(SRC_DIR)/tokenizer.cpp \
    $(SRC_DIR)/download.cpp

# Object files
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
SHARED_OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)

# CLI: main.cu with CLI main()
# Server: main.cu with -DQWEN_SERVER_BUILD (no CLI main) + server.cu
MAIN_CLI_OBJ = $(BUILD_DIR)/main.o
MAIN_SERVER_OBJ = $(BUILD_DIR)/main_server.o
SERVER_OBJ = $(BUILD_DIR)/server.o

CLI_TARGET = qwen-inference
SERVER_TARGET = qwen-server

.PHONY: all clean

all: $(CLI_TARGET) $(SERVER_TARGET)

$(CLI_TARGET): $(SHARED_OBJECTS) $(MAIN_CLI_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(SERVER_TARGET): $(SHARED_OBJECTS) $(MAIN_SERVER_OBJ) $(SERVER_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# main.cu for CLI (includes CLI main())
$(MAIN_CLI_OBJ): $(SRC_DIR)/main.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# main.cu for server (excludes CLI main())
$(MAIN_SERVER_OBJ): $(SRC_DIR)/main.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -DQWEN_SERVER_BUILD -c -o $@ $<

# server.cu (server main() + HTTP routes)
$(SERVER_OBJ): $(SRC_DIR)/server.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -DQWEN_SERVER_BUILD -c -o $@ $<

# CUDA source compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# C++ source compilation (compiled through nvcc for cuda_bf16.h)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -x cu -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) $(CLI_TARGET) $(SERVER_TARGET)
