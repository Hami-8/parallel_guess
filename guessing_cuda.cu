#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "PCFG.h"
#include <cuda_runtime.h> 

// -------------------------------------------------------
// 将 ordered_values 整体搬到 GPU，
// 每个线程把 prefix+suffix 拼好 -> 写回 flat char 缓冲.
// -------------------------------------------------------
__global__
void combine_kernel(const char* __restrict d_suffix_buf,
                    const int*  __restrict d_off,
                    const int*  __restrict d_len,
                    int                        N_suffix,
                    const char* __restrict d_prefix,
                    int         prefix_len,
                    char*       d_out,        // 每串固定 MAX_PWD bytes
                    int         stride)       // =MAX_PWD
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_suffix) return;

    char* dst = d_out + idx * stride;
    // copy prefix
    for(int k=0;k<prefix_len;k++) dst[k] = d_prefix[k];
    // copy suffix
    const char* src = d_suffix_buf + d_off[idx];
    int s_len = d_len[idx];
    for(int k=0;k<s_len;k++) dst[prefix_len+k] = src[k];
    // 终止符
    dst[prefix_len + s_len] = '\0';
}

// ------------------ 封装为两个 Host API ------------------
static const int MAX_PWD = 64;     // 足够 RockYou 中常见长度

// ① 单-segment PT
void GPUGenerateSingleSeg(segment* seg,
                           std::vector<std::string>& out_vec,
                           cudaStream_t s)
{
    int N = seg->ordered_values.size();
    // 1) 把 suffix 扁平化
    thrust::host_vector<char>  h_pool;
    thrust::host_vector<int>   h_off(N), h_len(N);
    size_t cursor = 0;
    for(int i=0;i<N;++i){
        h_off[i] = cursor;
        h_len[i] = seg->ordered_values[i].size();
        cursor  += h_len[i];
    }
    h_pool.resize(cursor);
    for(int i=0;i<N;++i){
        memcpy(h_pool.data()+h_off[i],
               seg->ordered_values[i].data(),
               h_len[i]);
    }
    // 2) 拷到 GPU
    thrust::device_vector<char> d_pool   = h_pool;
    thrust::device_vector<int>  d_off    = h_off;
    thrust::device_vector<int>  d_len    = h_len;
    // 3) 结果缓冲
    thrust::device_vector<char> d_out(N * MAX_PWD);
    // 4) launch
    int block=256, grid=(N+block-1)/block;
    combine_kernel<<<grid,block,0,s>>>(
        thrust::raw_pointer_cast(d_pool.data()),
        thrust::raw_pointer_cast(d_off.data()),
        thrust::raw_pointer_cast(d_len.data()),
        N,
        nullptr, 0,                              // prefix=NULL
        thrust::raw_pointer_cast(d_out.data()),
        MAX_PWD);
    // cudaDeviceSynchronize();
    // // 5) 拷回
    // thrust::host_vector<char> h_out = d_out;
    // 异步拷回
    std::size_t bytes = static_cast<std::size_t>(N) * MAX_PWD;
    thrust::host_vector<char> h_out(bytes);           // 若之前已声明可复用
    cudaMemcpyAsync(h_out.data(), thrust::raw_pointer_cast(d_out.data()),
                    bytes, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    out_vec.reserve(out_vec.size()+N);
    for(int i=0;i<N;++i)
        out_vec.emplace_back(h_out.data()+i*MAX_PWD);
}

// ② 多-segment PT —— 给定 prefix & last_seg
void GPUGenerateLastSeg(const std::string& prefix,
                        segment* last_seg,
                        std::vector<std::string>& out_vec,
                           cudaStream_t s)
{
    int N = last_seg->ordered_values.size();

    // prefix 上 GPU
    thrust::device_vector<char> d_prefix(prefix.begin(), prefix.end());
    int prefix_len = prefix.size();

    // suffix pool 同上
    thrust::host_vector<char> h_pool;
    thrust::host_vector<int>  h_off(N), h_len(N);
    size_t p=0;
    for(int i=0;i<N;++i){
        h_off[i]=p; h_len[i]=last_seg->ordered_values[i].size();
        p += h_len[i];
    }
    h_pool.resize(p);
    for(int i=0;i<N;++i)
        memcpy(h_pool.data()+h_off[i],
               last_seg->ordered_values[i].data(),
               h_len[i]);

    thrust::device_vector<char> d_pool = h_pool;
    thrust::device_vector<int>  d_off  = h_off;
    thrust::device_vector<int>  d_len  = h_len;
    thrust::device_vector<char> d_out(N*MAX_PWD);

    int block=256, grid=(N+block-1)/block;
    combine_kernel<<<grid,block,0,s>>>(
        thrust::raw_pointer_cast(d_pool.data()),
        thrust::raw_pointer_cast(d_off.data()),
        thrust::raw_pointer_cast(d_len.data()),
        N,
        thrust::raw_pointer_cast(d_prefix.data()),
        prefix_len,
        thrust::raw_pointer_cast(d_out.data()),
        MAX_PWD);
    // cudaDeviceSynchronize();

    // // copy-back
    // thrust::host_vector<char> h_out = d_out;
    // 异步拷回
    std::size_t bytes = static_cast<std::size_t>(N) * MAX_PWD;
    thrust::host_vector<char> h_out(bytes);
    cudaMemcpyAsync(h_out.data(), thrust::raw_pointer_cast(d_out.data()),
                    bytes, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    out_vec.reserve(out_vec.size()+N);
    for(int i=0;i<N;++i)
        out_vec.emplace_back(h_out.data()+i*MAX_PWD);
}
