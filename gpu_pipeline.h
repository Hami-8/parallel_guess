#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include "PCFG.h"

// ---------------------- task / result ----------------------
struct GPUTask {
    segment*                         seg;       // 单 segment 或最后段
    std::string                      prefix;    // 空 = 单段
    std::shared_ptr<std::vector<std::string>> dst; // GPU 生成后写入这里
};

struct GPUResult {
    std::shared_ptr<std::vector<std::string>> dst; // 已填充字符串
};

// ---------------------- 全局管道 ---------------------------
inline std::queue<GPUTask>  g_inQ;
inline std::queue<GPUResult> g_doneQ;
inline std::mutex            g_m_in, g_m_done;
inline std::condition_variable g_cv_in;
inline bool                  g_stop_gpu=false;
inline std::thread*          g_gpu_thread=nullptr;

void gpu_worker();
