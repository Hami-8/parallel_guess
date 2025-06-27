#ifdef USE_CUDA
#include "gpu_scheduler.h"
#include "PCFG.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>

extern std::atomic<long long> g_generate_us;

cudaStream_t stream0;
cudaEvent_t  evt;

static bool stream_inited=false;

/* 帮助函数：拼 prefix */
std::string build_prefix(PriorityQueue* Q, const PT& pt)
{
    std::string prefix;
    int seg_idx=0;
    for(int idx: pt.curr_indices){
        auto &m = Q->m;
        if(pt.content[seg_idx].type==1)
            prefix+=m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
        else if(pt.content[seg_idx].type==2)
            prefix+=m.digits [m.FindDigit (pt.content[seg_idx])].ordered_values[idx];
        else
            prefix+=m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
        if(++seg_idx == (int)pt.content.size()-1) break;
    }
    return prefix;
}

/* ---- 核心：把 batch 发到 GPU, 回传后写 guesses ---- */
void SubmitBatchAndWait(Batch& batch, PriorityQueue* Q)
{
    if(batch.tasks.empty()) return;
    if(!stream_inited){
        cudaStreamCreate(&stream0);
        cudaEventCreate(&evt); stream_inited=true;
    }
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    /* 简化实现：逐 task 调 kernel（代码易懂），
       也可一次 flatten 所有 task，再大 kernel */
    for(Task& t: batch.tasks){
        if(t.prefix.empty())
            GPUGenerateSingleSeg(t.last_seg, Q->guesses);
        else
            GPUGenerateLastSeg(t.prefix, t.last_seg, Q->guesses);

        Q->total_guesses += t.workload;
    }

    auto t1 = clk::now();
    g_generate_us += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    batch.clear();
}
#endif
