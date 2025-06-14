## MPI 编译说明

### 1. 编译指令

#### (1) MPI 编译指令

```
mpic++ -std=c++17 main_mpi.cpp train.cpp guessing_mpi.cpp md5.cpp -o main
mpic++ -std=c++17 main_mpi.cpp train.cpp guessing_mpi.cpp md5.cpp -o main -O1
mpic++ -std=c++17 main_mpi.cpp train.cpp guessing_mpi.cpp md5.cpp -o main -O2
```


#### (2) MPI+SIMD 编译指令

```
mpic++ -std=c++17 main_mpi_simd.cpp train.cpp guessing_mpi.cpp md5_SIMD.cpp -o main
mpic++ -std=c++17 main_mpi_simd.cpp train.cpp guessing_mpi.cpp md5_SIMD.cpp -o main -O1
mpic++ -std=c++17 main_mpi_simd.cpp train.cpp guessing_mpi.cpp md5_SIMD.cpp -o main -O2
```

### 2. 执行指令

执行，默认node=1，ppn=8，以8进程执行：

```
qsub qsub_mpi.sh
```

### 3. 修改进程数

在 qsub_mpi.sh 中调整

```shell
# ...... 其他代码  ......
#PBS -l nodes=1:ppn=8
# ...... 其他代码  ......
/usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main
```


### 4. 新增文件


MPI并行相关：

- guessing_mpi.cpp ： MPI 版本的 guessing。
- main_mpi.cpp ： MPI 版本的main函数。
- main_mpi_simd.cpp ： MPI+SIMD 版本的main函数。
- main_mpi_correctness.cpp ：验证MPI Cracked 数的main函数。

串行版本相关：

- main_serial.cpp ： 串行版本的main函数。
- main_serial_simd.cpp ： 串行+SIMD 版本的main函数。