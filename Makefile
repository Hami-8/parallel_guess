NVCC := nvcc  -std=c++17 -O0
CPP  := g++   -std=c++17 -O0
SRC  := main_cuda.cpp train.cpp guessing_cuda.cpp md5.cpp
CUDA := guessing_cuda.cu
INC  := -I.

# ---------- 目标 ----------
TARGET := main

# ---------- 默认目标 ----------
$(TARGET): $(SRC) $(CUDA)
	$(NVCC) -DUSE_CUDA $(INC) -o $@ $^

# ---------- 清理 ----------
.PHONY: clean
clean:
	rm -f $(TARGET)
