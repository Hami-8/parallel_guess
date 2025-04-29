## SIMD 编程作业说明

### 1. 编译指令：

串行代码编译指令不变。

并行代码编译指令如下：
```
g++ main.cpp train.cpp guessing.cpp md5_SIMD.cpp -o main
g++ main.cpp train.cpp guessing.cpp md5_SIMD.cpp -o main -O1
g++ main.cpp train.cpp guessing.cpp md5_SIMD.cpp -o main -O2
```

### 2. 新增文件：
- md5_SIMD.h：是对 md5.h 中MD5计算函数的并行化
- md5_SIMD.cpp：是对 md5.cpp 中 MD5Hash() 函数的并行化适配，并行化函数为 MD5Hash_SIMD()