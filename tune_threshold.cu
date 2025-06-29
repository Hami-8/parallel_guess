#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

// 编译
// nvcc -O2 -std=c++17 tune_threshold.cu -o tune_threshold
// 运行
// ./tune_threshold

constexpr int MAX_PWD = 64;            // 与正式实现保持一致
constexpr int BLOCK   = 256;

/* ---------------- GPU kernel（与正式版相同） ---------------- */
__global__
void combine_kernel(const char* d_pool, const int* d_off, const int* d_len,
                    int N, char* d_out, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    char* dst = d_out + idx * stride;
    const char* src = d_pool + d_off[idx];
    int len = d_len[idx];
    for (int k = 0; k < len; ++k) dst[k] = src[k];
    dst[len] = '\0';
}

/* ---------------- 生成随机字符串 ---------------- */
std::vector<std::string> make_random_pool(int N, int len = 8)
{
    static const char tbl[] =
        "abcdefghijklmnopqrstuvwxyz0123456789";
    static thread_local std::mt19937 rng(114514);
    std::uniform_int_distribution<int> pick(0, sizeof(tbl) - 2);

    std::vector<std::string> vec;
    vec.reserve(N);
    std::string s(len, ' ');
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < len; ++j) s[j] = tbl[pick(rng)];
        vec.push_back(s);
    }
    return vec;
}

/* ---------------- CPU 串行计时 ---------------- */
double bench_cpu(const std::vector<std::string>& pool)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    std::vector<std::string> dst;
    dst.reserve(pool.size());
    for (auto& s : pool) dst.emplace_back(s);
    auto t1 = clk::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count(); // µs
}

/* ---------------- GPU 计时 ---------------- */
double bench_gpu(const std::vector<std::string>& pool)
{
    int N = pool.size();
    /* H->D 预处理 */
    std::vector<int> h_off(N), h_len(N);
    size_t cursor = 0;
    for (int i = 0; i < N; ++i) {
        h_off[i] = cursor;
        h_len[i] = pool[i].size();
        cursor  += h_len[i];
    }
    std::vector<char> h_pool(cursor);
    for (int i = 0; i < N; ++i)
        memcpy(h_pool.data() + h_off[i], pool[i].data(), h_len[i]);

    thrust::device_vector<char> d_pool   = h_pool;
    thrust::device_vector<int>  d_off    = h_off;
    thrust::device_vector<int>  d_len    = h_len;
    thrust::device_vector<char> d_out(N * MAX_PWD);

    int grid = (N + BLOCK - 1) / BLOCK;

    cudaDeviceSynchronize();          // 防止前序干扰
    auto t0 = std::chrono::high_resolution_clock::now();

    combine_kernel<<<grid, BLOCK>>>(thrust::raw_pointer_cast(d_pool.data()),
                                    thrust::raw_pointer_cast(d_off.data()),
                                    thrust::raw_pointer_cast(d_len.data()),
                                    N,
                                    thrust::raw_pointer_cast(d_out.data()),
                                    MAX_PWD);
    cudaDeviceSynchronize();          // 包括 D2H 也要同步
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count(); // µs
}

/* ---------------- 主函数 ---------------- */
int main()
{
    std::vector<int> test_N = {128, 256, 512, 1024, 2048,
                               4096, 8192, 16384, 32768};

    std::cout << "N\tCPU_us\tGPU_us\tPrefer\n";
    int threshold = -1;

    for (int N : test_N) {
        auto pool = make_random_pool(N, 8);

        double t_cpu = bench_cpu(pool);
        double t_gpu = bench_gpu(pool);

        std::cout << N << '\t'
                  << t_cpu << '\t'
                  << t_gpu << '\t'
                  << ((t_gpu < t_cpu) ? "GPU" : "CPU") << '\n';

        if (threshold < 0 && t_gpu < t_cpu) threshold = N;
    }

    if (threshold < 0) threshold = test_N.back() * 2;  // GPU 永远慢
    std::cout << "\nRecommended CPU_THRESHOLD = "
              << threshold << " (first GPU faster)\n";
    return 0;
}
