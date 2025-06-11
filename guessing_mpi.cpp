#include "PCFG.h"
using namespace std;
#include <chrono>
#include <mpi.h> 
#include <numeric>   // std::accumulate
#include <cstring>   // std::memcpy
std::atomic<long long> g_generate_us{0};   // 微秒累计
std::atomic<long long> g_merge_us{0};






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

void PriorityQueue::GenerateSerial(PT pt)
{
    using namespace std::chrono;
    auto t_start = high_resolution_clock::now();      // ⟵ 开始
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
    auto t_end   = high_resolution_clock::now();      // ⟵ 结束
    g_generate_us += duration_cast<microseconds>(t_end - t_start).count();
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
/* root 专用：把最后一个 segment 的所有 value 划片给 worker 并收集结果
 * 假设外层保证 size >= 1；若 size==1 直接调用 GenerateSerial()
 */
void PriorityQueue::Generate(PT pt)
{
    using namespace std::chrono;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // 这里 rank 必然是 0
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size == 1) {           // 单进程退化为串行
        GenerateSerial(pt);
        return;
    }

    /*------------------------------------------------------------
     * 0. root 构造公共数据：prefix、最后一段的 ordered_values[]
     *-----------------------------------------------------------*/
    string prefix;                 // 若 PT 只有 1 段，它为空
    segment *last_seg = nullptr;   // 指向模型中最后 segment
    if(pt.content.size() == 1) {
        auto &sg = pt.content[0];
        if(sg.type==1) last_seg=&m.letters[m.FindLetter(sg)];
        if(sg.type==2) last_seg=&m.digits [m.FindDigit (sg)];
        if(sg.type==3) last_seg=&m.symbols[m.FindSymbol(sg)];
    } else {
        /* 拼除最后一段外的前缀 */
        for(size_t i=0;i<pt.content.size()-1;i++){
            int idx = pt.curr_indices[i];
            auto &sg = pt.content[i];
            if(sg.type==1) prefix += m.letters[m.FindLetter(sg)].ordered_values[idx];
            if(sg.type==2) prefix += m.digits [m.FindDigit (sg)].ordered_values[idx];
            if(sg.type==3) prefix += m.symbols[m.FindSymbol(sg)].ordered_values[idx];
        }
        /* 取最后一段 pointer */
        auto &sg = pt.content.back();
        if(sg.type==1) last_seg=&m.letters[m.FindLetter(sg)];
        if(sg.type==2) last_seg=&m.digits [m.FindDigit (sg)];
        if(sg.type==3) last_seg=&m.symbols[m.FindSymbol(sg)];
    }

    int N = last_seg->ordered_values.size();      // 循环总数

    /*------------------------------------------------------------
     * 1. 把 ordered_values 扁平成 “\0” 分隔的 char 缓冲 flatBuf
     *    并构造 offset[i] → 每个 value 起始偏移
     *-----------------------------------------------------------*/
    vector<char>  flatBuf;
    vector<int>   offset;               // N+1: 最后一个元素是总长
    offset.reserve(N+1);

    size_t pos = 0;
    for(const string &v : last_seg->ordered_values){
        offset.push_back(static_cast<int>(pos));
        pos += v.size() + 1;            // +1 记录 '\0'
    }
    offset.push_back(static_cast<int>(pos));      // 末尾
    flatBuf.resize(pos);

    /* 填充 flatBuf */
    {
        char *p = flatBuf.data();
        for(const string &v : last_seg->ordered_values){
            memcpy(p, v.c_str(), v.size()+1);
            p += v.size()+1;
        }
    }

    /*------------------------------------------------------------
     * 2. 广播任务给所有进程
     *    顺序：N → prefix → flatBuf → offset
     *-----------------------------------------------------------*/
    auto bcast_str = [](string &s){
        int len = s.size();
        MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);
        if(len){
            if(MPI::COMM_WORLD.Get_rank()!=0) s.resize(len);
            MPI_Bcast(s.data(),len,MPI_CHAR,0,MPI_COMM_WORLD);
        }
    };

    MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);         // (a) N
    bcast_str(prefix);                                 // (b) prefix

    int bufLen = flatBuf.size();
    MPI_Bcast(&bufLen,1,MPI_INT,0,MPI_COMM_WORLD);     // (c.1) flatBuf 长度
    MPI_Bcast(flatBuf.data(),bufLen,MPI_CHAR,0,MPI_COMM_WORLD); // (c.2) 实际内容

    int offLen = offset.size();
    MPI_Bcast(&offLen,1,MPI_INT,0,MPI_COMM_WORLD);     // (d.1) offset 长度
    MPI_Bcast(offset.data(),offLen,MPI_INT,0,MPI_COMM_WORLD);   // (d.2) offset 内容

    /*------------------------------------------------------------
     * 3. root 自己也生成一片：按“等块划分”策略
     *-----------------------------------------------------------*/
    int base  = N / size;
    int extra = N % size;
    int start = 0 * base + std::min(0,extra);
    int end   = start + base + (0 < extra);

    high_resolution_clock::time_point t_start = high_resolution_clock::now();

    for(int i = start; i < end; ++i){
        const char* val = flatBuf.data() + offset[i];
        string guess = prefix + string(val);
        guesses.emplace_back(std::move(guess));
        total_guesses++;
    }

    /*------------------------------------------------------------
     * 4. 接收每个 worker 的结果 (bytes + payload)
     *-----------------------------------------------------------*/
    for(int src=1; src<size; ++src){
        int bytes = 0;
        MPI_Recv(&bytes,1,MPI_INT,src,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(bytes==0) continue;
        vector<char> buf(bytes);
        MPI_Recv(buf.data(),bytes,MPI_CHAR,src,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        /* 把 buf 拆成字符串 */
        char *p = buf.data();
        while(p < buf.data()+bytes){
            string g(p);
            guesses.emplace_back(std::move(g));
            p += g.size()+1;
            total_guesses++;
        }
    }

    auto t_end = high_resolution_clock::now();
    g_generate_us += duration_cast<microseconds>(t_end - t_start).count();
}