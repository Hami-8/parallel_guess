#include <cuda_runtime.h>
#include "PCFG.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// ①  kernel：每线程 1 口令
__global__ void combine_kernel(
        const char* __restrict d_suffix_pool,
        const int*  __restrict d_off,
        const int*  __restrict d_len,
        int N_suffix,
        const char* __restrict d_prefix,
        int prefix_len,
        char* d_out,
        int stride)                       // MAX_PWD
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_suffix) return;

    char* dst = d_out + (size_t)idx * stride;

    // copy prefix
    for(int k=0;k<prefix_len;k++) dst[k] = d_prefix[k];

    // copy suffix
    const char* src = d_suffix_pool + d_off[idx];
    int s_len = d_len[idx];
    for(int k=0;k<s_len;k++) dst[prefix_len+k] = src[k];

    dst[prefix_len + s_len] = '\0';
}

/* helper：单 segment */
void GPUGenerateSingleSeg(segment* seg, std::vector<std::string>& out_vec)
{
    int N = seg->ordered_values.size();
    if(N==0) return;

    // ------- flatten suffix -----------
    std::vector<int> h_off(N), h_len(N);
    std::vector<char> h_pool;
    size_t cursor=0;
    for(int i=0;i<N;i++){
        h_off[i]=cursor;
        h_len[i]=seg->ordered_values[i].size();
        cursor += h_len[i];
    }
    h_pool.resize(cursor);
    for(int i=0;i<N;i++)
        memcpy(&h_pool[h_off[i]], seg->ordered_values[i].data(), h_len[i]);

    // ------- upload -----------
    thrust::device_vector<char> d_pool = h_pool;
    thrust::device_vector<int>  d_off  = h_off;
    thrust::device_vector<int>  d_len  = h_len;
    thrust::device_vector<char> d_out ((size_t)N*MAX_PWD);

    int block=256, grid=(N+block-1)/block;
    combine_kernel<<<grid,block>>>(
        thrust::raw_pointer_cast(d_pool.data()),
        thrust::raw_pointer_cast(d_off.data()),
        thrust::raw_pointer_cast(d_len.data()),
        N, nullptr, 0,
        thrust::raw_pointer_cast(d_out.data()),
        MAX_PWD);
    cudaDeviceSynchronize();

    // ------- download -----------
    thrust::host_vector<char> h_out = d_out;
    out_vec.reserve(out_vec.size()+N);
    for(int i=0;i<N;i++)
        out_vec.emplace_back(h_out.data()+ (size_t)i*MAX_PWD);
}

/* helper：已知 prefix + last_seg */
void GPUGenerateLastSeg(const std::string& prefix,
                        segment* last_seg,
                        std::vector<std::string>& out_vec)
{
    int N = last_seg->ordered_values.size();
    if(N==0) return;

    // prefix 到设备
    thrust::device_vector<char> d_pre(prefix.begin(), prefix.end());
    int pre_len = prefix.size();

    // suffix pool
    std::vector<int> h_off(N), h_len(N);
    std::vector<char> h_pool;
    size_t cur=0;
    for(int i=0;i<N;i++){
        h_off[i]=cur;
        h_len[i]=last_seg->ordered_values[i].size();
        cur += h_len[i];
    }
    h_pool.resize(cur);
    for(int i=0;i<N;i++)
        memcpy(&h_pool[h_off[i]], last_seg->ordered_values[i].data(), h_len[i]);

    thrust::device_vector<char> d_pool = h_pool;
    thrust::device_vector<int>  d_off  = h_off;
    thrust::device_vector<int>  d_len  = h_len;
    thrust::device_vector<char> d_out ((size_t)N*MAX_PWD);

    int block=256, grid=(N+block-1)/block;
    combine_kernel<<<grid,block>>>(
        thrust::raw_pointer_cast(d_pool.data()),
        thrust::raw_pointer_cast(d_off.data()),
        thrust::raw_pointer_cast(d_len.data()),
        N,
        thrust::raw_pointer_cast(d_pre.data()),
        pre_len,
        thrust::raw_pointer_cast(d_out.data()),
        MAX_PWD);
    cudaDeviceSynchronize();

    thrust::host_vector<char> h_out = d_out;
    out_vec.reserve(out_vec.size()+N);
    for(int i=0;i<N;i++)
        out_vec.emplace_back(h_out.data()+ (size_t)i*MAX_PWD);
}
