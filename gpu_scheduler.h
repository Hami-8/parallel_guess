#pragma once
#include <vector>
#include <string>
#include "PCFG.h"

#ifdef USE_CUDA
struct Task {
    std::string prefix;
    segment*    last_seg;
    int         workload;        // == suffix count
};
struct Batch {
    std::vector<Task> tasks;
    int total_pwd = 0;
    void clear() { tasks.clear(); total_pwd=0; }
};
std::string build_prefix(PriorityQueue* Q, const PT& pt);
void SubmitBatchAndWait(Batch& batch, class PriorityQueue* Q);
#endif
