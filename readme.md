## CUDA 编译说明

### 1. 编译运行指令

编译脚本已经写在了Makefile里，默认以 -O2 优化等级运行：

```
make clean
make -j
```

运行：

```
./main      > test.o   2> test.e
```

若要修改优化等级，可修改Makefile的前两行：

```
NVCC := nvcc  -std=c++17 -O2
CPP  := g++   -std=c++17 -O2
```

### 2. 不同 CUDA 并行版本



#### 2.1 基础 CUDA 并行版本

基础CUDA版本在git的cuda_basic分支，切换到该分支运行即可
```
git checkout cuda_basic
make clean
make -j
./main      > test.o   2> test.e
```

#### 2.2 CPU-GPU 静态自适应调度版本

CPU-GPU 静态自适应调度版本在git的cuda_static_scheduling分支：

```
git checkout cuda_static_scheduling
make clean
make -j
./main      > test.o   2> test.e
```

#### 2.3 CPU-GPU 动态自适应调度版本

CPU-GPU 动态自适应调度版本在git的cuda_dynamic_scheduling分支：

```
git checkout cuda_dynamic_scheduling
make clean
make -j
./main      > test.o   2> test.e
```

#### 2.4 CPU-GPU 流水并行版本

CPU-GPU 流水并行版本在git的main分支：

```
git checkout main
make clean
make -j
./main      > test.o   2> test.e
```

### 3.其他脚本

#### tune_threshold.cu

是**静态自适应调度**的离线微基准测试脚本，编译运行命令：

```
nvcc -O2 -std=c++17 tune_threshold.cu -o tune_threshold
./tune_threshold
```

#### measure.sh

硬件利用率测试脚本，先提权，再运行：

```
chmod +x measure.sh
./measure.sh
```

### 4. 新增文件

- guessing_cuda.cpp ：cuda版本的guessing
- guessing_cuda.cu : 线程核函数代码
- gpu_pipeline.h ：CPU-GPU 流水并行的相关数据结构
- main_cuda.cpp ： cuda版本的main函数
- main_cuda_correctness.cpp ：检验cuda版本正确性的main函数
- Makefile：编译脚本
- measure.sh ：硬件利用率测试脚本
- tune_threshold.cu ：静态自适应调度的离线微基准测试脚本