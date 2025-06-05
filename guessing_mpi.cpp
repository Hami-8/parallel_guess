#include "PCFG.h"
using namespace std;
#include <chrono>
#include <mpi.h> 
#include <numeric>   // std::accumulate
#include <cstring>   // std::memcpy
std::atomic<long long> g_generate_us{0};   // 微秒累计
std::atomic<long long> g_merge_us{0};
static MPI_Comm pw_comm = MPI_COMM_WORLD;


void Generate_MPI_Worker(int maxN)
{
    /* 此函数执行下列步骤：
       1) 已经拿到 maxN —— 就按和 root 同样的 Bcast 顺序
          再连续 MPI_Bcast prefix / lens[] / flat_values
       2) 划分 rank 自己的 [start,end) 区间
       3) 拼接本地口令（可直接丢弃或统计条数即可）
    */

    int rank, size; 
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    /* --- 接收 prefix --- */
    int plen;  MPI_Bcast(&plen,1,MPI_INT,0,MPI_COMM_WORLD);
    std::string prefix(plen,'\0');
    if (plen) MPI_Bcast(prefix.data(), plen, MPI_CHAR, 0, MPI_COMM_WORLD);

    /* --- 接收 lens[] 与扁平值块 --- */
    std::vector<int> lens(maxN);
    MPI_Bcast(lens.data(), maxN, MPI_INT, 0, MPI_COMM_WORLD);

    int flatLen; MPI_Bcast(&flatLen,1,MPI_INT,0,MPI_COMM_WORLD);
    std::string flat(flatLen,'\0');
    if (flatLen) MPI_Bcast(flat.data(), flatLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    /* --- 切割 value 字符串 --- */
    std::vector<std::string> segVals(maxN);
    int off=0;
    for(int i=0;i<maxN;++i){
        segVals[i].assign(flat.data()+off, (size_t)lens[i]);
        off += lens[i];
    }

    /* --- 计算工作区间并生成 (可省内存只统计) --- */
    int base = maxN / size, extra = maxN % size;
    int start = rank*base + std::min(rank,extra);
    int end   = start + base + (rank<extra);

    for(int i=start;i<end;++i){
        /* 若只需算 MD5、或只需返回条数，这里可省内存 */
        volatile std::string tmp = prefix + segVals[i];
        (void)tmp;      // 占位，防优化
    }
}


/* 被动服务：不停接收 Generate_MPI 的广播，直至 stop_flag=-1 */
void WorkerLoop()
{
    int flag;
    while (true)
    {
        /* 先探测是否有“停止”广播 */
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (flag == -1) break;                 // 主进程宣告任务完成

        /* —— 其余情况：flag 就是 maxN —— */
        /* 我们需要把它推回缓冲，让 Generate_MPI_Worker() 能正常取到 —— */
        /* 方案：用 MPI_Bcast 'MPI_IN_PLACE' 回填，或直接调用一个只做“接收” */
        Generate_MPI_Worker(flag);             
    }
}



// ────────────────────────────────────────────────────────────────
// 把每个进程的 vector<string> 聚合到 root
// root_buf 只在 root 内部填充，其余进程可传一个 dummy 变量
// ────────────────────────────────────────────────────────────────
void gather_strings_to_root(const std::vector<std::string>& local,
                            int root, MPI_Comm comm,
                            std::vector<std::string>& root_buf)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* ----------------------------------------------------------
     * ① 先收集“每个进程有多少条字符串”
     * -------------------------------------------------------- */
    int local_count = static_cast<int>(local.size());
    std::vector<int> all_counts(size);        // 仅 root 用到
    MPI_Gather(&local_count, 1, MPI_INT,
               all_counts.data(), 1, MPI_INT,
               root, comm);

    /* ----------------------------------------------------------
     * ② 收集每条字符串长度（变长数据，需要 Gatherv）
     * -------------------------------------------------------- */
    std::vector<int> local_lens(local_count);
    for (int i = 0; i < local_count; ++i)
        local_lens[i] = static_cast<int>(local[i].size());

    // 统计全局元素总数 (root 端)
    int total_strings = 0;
    if (rank == root)
        total_strings = std::accumulate(all_counts.begin(),
                                        all_counts.end(), 0);

    // root 预分配长度数组
    std::vector<int> all_lens;
    std::vector<int> lens_disp;
    if (rank == root) {
        all_lens.resize(total_strings);
        lens_disp.resize(size, 0);
        for (int i = 1; i < size; ++i)
            lens_disp[i] = lens_disp[i - 1] + all_counts[i - 1];
    }

    MPI_Gatherv(local_lens.data(),            // sendbuf
                local_count, MPI_INT,
                /*root recv*/ (rank==root ? all_lens.data() : nullptr),
                /*recvcounts*/ (rank==root ? all_counts.data() : nullptr),
                /*displs*/     (rank==root ? lens_disp.data() : nullptr),
                MPI_INT, root, comm);

    /* ----------------------------------------------------------
     * ③ 收集真正的字节内容（再次 Gatherv）
     * -------------------------------------------------------- */
    // 把本地字符串扁平化
    int local_bytes = 0;
    for (const auto& s : local) local_bytes += static_cast<int>(s.size());

    std::vector<char> byte_buf(local_bytes);
    int offset = 0;
    for (const auto& s : local) {
        std::memcpy(byte_buf.data() + offset, s.data(), s.size());
        offset += static_cast<int>(s.size());
    }

    // root 准备全局缓冲
    std::vector<int> bytes_counts(size), bytes_disp(size, 0);
    MPI_Gather(&local_bytes, 1, MPI_INT,
               bytes_counts.data(), 1, MPI_INT,
               root, comm);

    int total_bytes = 0;
    if (rank == root) {
        total_bytes = std::accumulate(bytes_counts.begin(),
                                      bytes_counts.end(), 0);
        for (int i = 1; i < size; ++i)
            bytes_disp[i] = bytes_disp[i - 1] + bytes_counts[i - 1];
    }

    std::vector<char> all_bytes;
    if (rank == root) all_bytes.resize(total_bytes);

    MPI_Gatherv(byte_buf.data(), local_bytes, MPI_CHAR,
                (rank==root ? all_bytes.data() : nullptr),
                (rank==root ? bytes_counts.data() : nullptr),
                (rank==root ? bytes_disp.data() : nullptr),
                MPI_CHAR, root, comm);

    /* ----------------------------------------------------------
     * ④ root 端重建 vector<string>
     * -------------------------------------------------------- */
    if (rank == root) {
        root_buf.clear();
        root_buf.reserve(total_strings);

        int cur_byte = 0;
        for (int len : all_lens) {
            root_buf.emplace_back(all_bytes.data() + cur_byte,
                                   static_cast<size_t>(len));
            cur_byte += len;
        }
    }
}

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
    using namespace std::chrono;
    int rank, size;
    MPI_Comm_rank(pw_comm, &rank);
    MPI_Comm_size(pw_comm, &size);

    /* 单进程时直接退化为串行 Generate --------------------------- */
    // if (size == 1) { Generate(pt); return; }

    auto t0 = high_resolution_clock::now();

    if (rank == 0) CalProb(pt);          // 概率初始化仅 root 需要

    /******** 1. root 构造 prefix、最后一个 segment 的 value 列表 ********/
    segment *last_seg = nullptr;
    std::string prefix;                  // 若 PT 只有 1 段，则为空串
    int maxN = 0;                        // value 个数

    if (rank == 0)
    {
        /* ① 找到最后一个 segment 在模型中的指针 ------------------- */
        auto locate_seg_ptr = [&](segment &sg) -> segment* {
            if (sg.type == 1) return &m.letters[m.FindLetter(sg)];
            if (sg.type == 2) return &m.digits [m.FindDigit (sg)];
            return            &m.symbols[m.FindSymbol(sg)];
        };

        // 只有 1 个 segment
        if (pt.content.size() == 1)
        {
            last_seg = locate_seg_ptr(pt.content[0]);
        }
        else
        {
            /* ② 先拼 prefix（除最后一段外的各 segment value） ------ */
            int idx_seg = 0;
            for (int idx_val : pt.curr_indices)
            {
                if (idx_seg == pt.content.size() - 1) break;
                segment &sg = pt.content[idx_seg];
                if (sg.type == 1)
                    prefix += m.letters[m.FindLetter(sg)].ordered_values[idx_val];
                if (sg.type == 2)
                    prefix += m.digits[m.FindDigit(sg)].ordered_values[idx_val];
                if (sg.type == 3)
                    prefix += m.symbols[m.FindSymbol(sg)].ordered_values[idx_val];
                ++idx_seg;
            }
            last_seg = locate_seg_ptr(pt.content.back());
        }
        maxN = pt.max_indices.back();    // 最后一段 value 总数
    }

    /******** 2. 广播公共参数：maxN、prefix、value 列表 ********/
    MPI_Bcast(&maxN, 1, MPI_INT, 0, pw_comm);

    /* ---- prefix ---- */
    int plen = (rank == 0) ? static_cast<int>(prefix.size()) : 0;
    MPI_Bcast(&plen, 1, MPI_INT, 0, pw_comm);
    if (rank != 0) prefix.resize(plen);
    void *prefix_buf = (plen == 0) ? nullptr
                                   : const_cast<char *>(prefix.data());
    MPI_Bcast(prefix_buf, plen, MPI_CHAR, 0, pw_comm);

    /* ---- value 列表：先广播长度数组，再广播扁平字符块 ---- */
    std::vector<int> lens(maxN);          // 每个 value 的字符长度
    std::string      flat_values;         // 所有 value 拼接

    if (rank == 0)
    {
        lens.reserve(maxN);
        for (int i = 0; i < maxN; ++i)
        {
            const std::string &v = last_seg->ordered_values[i];
            lens[i] = static_cast<int>(v.size());
            flat_values += v;
        }
    }

    MPI_Bcast(lens.data(), maxN, MPI_INT, 0, pw_comm);

    int flatLen = (rank == 0) ? static_cast<int>(flat_values.size()) : 0;
    MPI_Bcast(&flatLen, 1, MPI_INT, 0, pw_comm);
    if (rank != 0) flat_values.resize(flatLen);
    void *flat_buf = (flatLen == 0) ? nullptr
                                    : const_cast<char *>(flat_values.data());
    MPI_Bcast(flat_buf, flatLen, MPI_CHAR, 0, pw_comm);

    /* 非 root 分割 flat_values → segVals --------------------- */
    std::vector<std::string> segVals(maxN);
    int offset = 0;
    for (int i = 0; i < maxN; ++i)
    {
        segVals[i].assign(flat_values.data() + offset,
                          static_cast<size_t>(lens[i]));
        offset += lens[i];
    }

    /******** 3. 均匀划分 value 下标区间 & 本地拼接 ********/
    int base  = maxN / size;
    int extra = maxN % size;
    int start = rank * base + std::min(rank, extra);
    int end   = start + base + (rank < extra);

    std::vector<std::string> local;
    local.reserve(end - start);

    for (int i = start; i < end; ++i)
        local.emplace_back(prefix + segVals[i]);

    /******** 4. 回收到 root，并统计 g_generate_us ********/
    std::vector<std::string> root_buf;      // 仅 root 写入
    gather_strings_to_root(local, 0, pw_comm, root_buf);

    if (rank == 0)
    {
        guesses.insert(guesses.end(),
                       root_buf.begin(), root_buf.end());
        total_guesses += static_cast<int>(root_buf.size());

        auto t1 = high_resolution_clock::now();
        g_generate_us += duration_cast<microseconds>(t1 - t0).count();
    }

    /* 其余 rank 直接返回 */
}