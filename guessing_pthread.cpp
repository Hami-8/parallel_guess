#include "PCFG.h"

#include <pthread.h>
#include <atomic>

// ---------------------------- 线程配置 -----------------------------
#ifndef THREAD_NUM
#define THREAD_NUM 7            // 可在编译命令中 -DTHREAD_NUM=N 动态修改
#endif

// --------------------- 帮助数据结构和函数 -------------------------
namespace {

struct ThreadArg {
    int  tid;                  // 线程编号
    int  begin;                // 处理区间起始下标（包含）
    int  end;                  // 处理区间结束下标（不含）
    const std::string *prefix; // 已经拼接好的前缀字符串 (可为空字符串)
    const segment      *seg;   // 指向最后一个 segment 的统计数据
    std::vector<std::string> *local_out; // 线程局部输出容器
};

void *generate_worker(void *arg_ptr) {
    ThreadArg *arg = static_cast<ThreadArg *>(arg_ptr);
    const std::string &pre = *arg->prefix;
    const segment    *s   = arg->seg;
    std::vector<std::string> &out = *arg->local_out;
    out.reserve(arg->end - arg->begin);

    for (int i = arg->begin; i < arg->end; ++i) {
        out.emplace_back(pre + s->ordered_values[i]);
    }
    return nullptr;
}

} // anonymous namespace
// using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);   // 初始概率

    // ---------- Case A: 只有一个 segment ----------
    if (pt.content.size() == 1)
    {
        const segment *a = nullptr;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];

        const int total = pt.max_indices[0];
        // if (total < 4000)
        // { // 太小直接串行
        //     for (int i = 0; i < pt.max_indices[0]; i += 1)
        //     {
        //         string guess = a->ordered_values[i];
        //         // cout << guess << endl;
        //         guesses.emplace_back(guess);
        //         total_guesses += 1;
        //     }
        //     return;
        // }
        const int chunk = (total + THREAD_NUM - 1) / THREAD_NUM;

        pthread_t threads[THREAD_NUM];
        ThreadArg targs[THREAD_NUM];
        std::vector<std::string> local_out[THREAD_NUM];

        std::string empty_prefix;
        for (int t = 0; t < THREAD_NUM; ++t)
        {
            int L = t * chunk;
            int R = std::min(total, L + chunk);
            // cout<<"线程 "<<t<<" 的 L:"<<L<<", R:"<<R<<endl;
            if (L >= R) { 
                // 让它处理一个空区间
                targs[t] = {t, 0, 0, &empty_prefix, a, &local_out[t]};
            }
            else{
            targs[t] = {t, L, R, &empty_prefix, a, &local_out[t]};
        }
            pthread_create(&threads[t], nullptr, generate_worker, &targs[t]);
        }
        for (int t = 0; t < THREAD_NUM; ++t) pthread_join(threads[t], nullptr);

        // 合并线程输出
        for (int t = 0; t < THREAD_NUM; ++t)
            guesses.insert(guesses.end(), local_out[t].begin(), local_out[t].end());

        total_guesses += total;
        return;
    }

    // ---------- Case B: 多个 segment，最后一个待填充 ----------

    // 1) 构造前缀 (除最后一个 segment 外)
    std::string prefix;
    int seg_idx = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[seg_idx].type == 1)
            prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
        if (pt.content[seg_idx].type == 2)
            prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
        if (pt.content[seg_idx].type == 3)
            prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
        ++seg_idx;
        if (seg_idx == pt.content.size() - 1) break; // 留出最后一个 segment
    }

    // 2) 定位最后一个 segment 的统计数据
    const segment *last_seg = nullptr;
    const segment &seg_obj = pt.content.back();
    if (seg_obj.type == 1) last_seg = &m.letters[m.FindLetter(seg_obj)];
    if (seg_obj.type == 2) last_seg = &m.digits[m.FindDigit(seg_obj)];
    if (seg_obj.type == 3) last_seg = &m.symbols[m.FindSymbol(seg_obj)];

    const int total = pt.max_indices.back();
        // if (total < 4000)
        // { // 太小直接串行
        //     for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        //     {
        //         guesses.emplace_back(prefix + last_seg->ordered_values[i]);
        //         total_guesses += 1;
        //     }
        //     return;
        // }
    const int chunk = (total + THREAD_NUM - 1) / THREAD_NUM;

    pthread_t threads[THREAD_NUM];
    ThreadArg targs[THREAD_NUM];
    std::vector<std::string> local_out[THREAD_NUM];

    for (int t = 0; t < THREAD_NUM; ++t)
    {
        int L = t * chunk;
        int R = std::min(total, L + chunk);
        // cout<<"线程 "<<t<<" 的 L:"<<L<<", R:"<<R<<endl;
        if (L >= R) { // 让它处理一个空区间
            targs[t] = {t, 0, 0, &prefix, last_seg, &local_out[t]}; 
        }
        else{
        targs[t] = {t, L, R, &prefix, last_seg, &local_out[t]};
    }
        pthread_create(&threads[t], nullptr, generate_worker, &targs[t]);
    }
    for (int t = 0; t < THREAD_NUM; ++t) pthread_join(threads[t], nullptr);

    // 3) 合并输出
    for (int t = 0; t < THREAD_NUM; ++t)
        guesses.insert(guesses.end(), local_out[t].begin(), local_out[t].end());

    total_guesses += total;
}