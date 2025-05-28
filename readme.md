## Pthread & Openmp 多线程编译说明

### 1. 编译指令

#### (1) 串行代码编译指令

main_serial.cpp 是串行的main文件。

```
g++ main_serial.cpp train.cpp guessing.cpp md5.cpp -o main
g++ main_serial.cpp train.cpp guessing.cpp md5.cpp -o main -O1
g++ main_serial.cpp train.cpp guessing.cpp md5.cpp -o main -O2
```

#### (2) 串行+SIMD 编译指令

main_serial_simd.cpp 是串行+SIMD的main文件。

```
g++ main_serial_simd.cpp train.cpp guessing.cpp md5_SIMD.cpp -o main
g++ main_serial_simd.cpp train.cpp guessing.cpp md5_SIMD.cpp -o main -O1
g++ main_serial_simd.cpp train.cpp guessing.cpp md5_SIMD.cpp -o main -O2
```



#### (3). Pthread 编译指令

main.cpp 是pthread和Openmp的main文件。


```
g++ main.cpp train.cpp guessing_pthread.cpp md5.cpp -o main -pthread
g++ main.cpp train.cpp guessing_pthread.cpp md5.cpp -o main -pthread -O1
g++ main.cpp train.cpp guessing_pthread.cpp md5.cpp -o main -pthread -O2
```

#### (4) Openmp 编译指令

```
g++ main.cpp train.cpp guessing_openmp.cpp md5.cpp -o main -fopenmp
g++ main.cpp train.cpp guessing_openmp.cpp md5.cpp -o main -fopenmp -O1
g++ main.cpp train.cpp guessing_openmp.cpp md5.cpp -o main -fopenmp -O2
```

#### (5) Pthread+SIMD 编译指令

main_simd.cpp 是多线程+SIMD的main文件。

```
g++ main_simd.cpp train.cpp guessing_pthread.cpp md5_SIMD.cpp -o main -pthread
g++ main_simd.cpp train.cpp guessing_pthread.cpp md5_SIMD.cpp -o main -pthread -O1
g++ main_simd.cpp train.cpp guessing_pthread.cpp md5_SIMD.cpp -o main -pthread -O2
```

#### (6) Openmp+SIMD 编译指令

```
g++ main_simd.cpp train.cpp guessing_openmp.cpp md5_SIMD.cpp -o main -fopenmp
g++ main_simd.cpp train.cpp guessing_openmp.cpp md5_SIMD.cpp -o main -fopenmp -O1
g++ main_simd.cpp train.cpp guessing_openmp.cpp md5_SIMD.cpp -o main -fopenmp -O2
```
### 2. 修改线程数

在 PCFG.h 的 class PriorityQueue 中

```cpp
/* 并行相关 ↓ */
    int                           num_threads = 4;      // 线程数，默认为4
```

### 3. 新增文件

- guessing_pthread.cpp ：Pthread 版本的 guessing。
- guessing_openmp.cpp ：Openmp 版本的 guessing。
- main_serial.cpp ： 串行版本的main函数。
- main_serial_simd.cpp ： 串行+SIMD 版本的main函数。
- main_simd.cpp ： 多线程+SIMD 版本的main函数。