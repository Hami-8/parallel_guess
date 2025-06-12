#include "PCFG.h"
using namespace std;
#include <chrono>
#include <mpi.h> 
#include <numeric>   // std::accumulate
#include <cstring>   // std::memcpy
std::atomic<long long> g_generate_us{0};   // 微秒累计
std::atomic<long long> g_merge_us{0};



const int MAX_SEG = 8;      // 假定 PT 最多 8 段


#pragma pack(push,1)
struct PackedPT {
    uint8_t seg_cnt;
    uint8_t type [MAX_SEG];
    uint8_t len  [MAX_SEG];
    uint16_t curr_idx[MAX_SEG];
    uint16_t max_idx [MAX_SEG];
    float   pre_prob;
};
#pragma pack(pop)

/* ---------- 辅助函数 ---------- */
static void PackPT(const PT& src, PackedPT& dst){
    dst.seg_cnt = src.content.size();
    for(int i=0;i<dst.seg_cnt;i++){
        dst.type[i] = src.content[i].type;
        dst.len [i] = src.content[i].length;
        dst.max_idx[i] = src.max_indices[i];
    }
    /* 最后一个段不需要 curr_idx */
    for(int i=0;i<dst.seg_cnt-1;i++) dst.curr_idx[i] = src.curr_indices[i];
    dst.pre_prob = src.preterm_prob;
}

static PT UnpackPT(const PackedPT& p, model &m){
    PT pt;
    for(int i=0;i<p.seg_cnt;i++){
        segment s(p.type[i],p.len[i]);
        pt.content.push_back(s);
        pt.max_indices.push_back(p.max_idx[i]);
    }
    for(int i=0;i<p.seg_cnt-1;i++) pt.curr_indices.push_back(p.curr_idx[i]);
    pt.preterm_prob = p.pre_prob;
    return pt;
}


// 解析一段 recvBuf -> 依次生成明文口令; 返回生成数量
int process_PT_buffer(std::vector<uint8_t>& buf,model& g_model){
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    model &mdl = g_model;      // 获取全局模型引用
    int total = 0;

    size_t pos = 0;
    while(pos < buf.size()){
        PackedPT *p = reinterpret_cast<PackedPT*>(buf.data()+pos);
        PT  pt = UnpackPT(*p, mdl);

        /* 调用已有串行生成函数，但写进局部 dummy 向量即可 */
        PriorityQueue dummy; dummy.m = mdl;
        dummy.GenerateSerial(pt);       // 把生成代码复用
        total += dummy.total_guesses;

        pos += sizeof(PackedPT);
    }
    return total;
}

inline void SerializeModel(const model &m, std::vector<uint8_t>& buf) {
    auto dump_seg_vec = [&](const std::vector<segment>& vec){
        uint32_t n = vec.size();
        buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&n),
                   reinterpret_cast<uint8_t*>(&n)+4);
        for(const auto& s: vec){
            buf.push_back(static_cast<uint8_t>(s.type));
            buf.push_back(static_cast<uint8_t>(s.length));
            uint32_t cnt = s.ordered_values.size();
            buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&cnt),
                       reinterpret_cast<uint8_t*>(&cnt)+4);
            for(const std::string& v: s.ordered_values){
                uint16_t len = v.size();
                buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&len),
                           reinterpret_cast<uint8_t*>(&len)+2);
                buf.insert(buf.end(), v.begin(), v.end());
            }
        }
    };
    dump_seg_vec(m.letters);
    dump_seg_vec(m.digits);
    dump_seg_vec(m.symbols);
}

inline void DeserializeModel(const uint8_t* p, model& m){
    auto load_seg_vec = [&](std::vector<segment>& vec){
        uint32_t n; memcpy(&n,p,4); p+=4;
        vec.reserve(n);
        for(uint32_t i=0;i<n;i++){
            uint8_t tp = *p++; uint8_t ln = *p++;
            uint32_t cnt; memcpy(&cnt,p,4); p+=4;
            segment s(tp,ln);
            s.ordered_values.reserve(cnt);
            // s.max_indices.reserve(cnt);
            for(uint32_t j=0;j<cnt;j++){
                uint16_t slen; memcpy(&slen,p,2); p+=2;
                std::string v(reinterpret_cast<const char*>(p),slen);
                p+=slen;
                s.ordered_values.push_back(std::move(v));
            }
            // s.max_indices.emplace_back(cnt);
            vec.emplace_back(std::move(s));
        }
    };
    load_seg_vec(m.letters);
    load_seg_vec(m.digits);
    load_seg_vec(m.symbols);
}

void BroadcastModel(model& m) {          // root 调用
    std::vector<uint8_t> buf;
    SerializeModel(m,buf);
    uint32_t len = buf.size();
    MPI_Bcast(&len,1,MPI_UINT32_T,0,MPI_COMM_WORLD);
    MPI_Bcast(buf.data(),len,MPI_BYTE,0,MPI_COMM_WORLD);
}

void FetchModel(model& m){               // worker 调用
    uint32_t len;
    MPI_Bcast(&len,1,MPI_UINT32_T,0,MPI_COMM_WORLD);
    std::vector<uint8_t> buf(len);
    MPI_Bcast(buf.data(),len,MPI_BYTE,0,MPI_COMM_WORLD);
    DeserializeModel(buf.data(),m);
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
GenerateSerial(pt);      // 仅保留串行以供内部复用
}


void PriorityQueue::BatchGenerateMPI(int K)     // K = 每 worker PT 数
{
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(size==1){    // 退化串行
        PopNext();  return;
    }

    /*---------------- root 逻辑 ----------------*/
    if(rank==0){
        int workers = size;          // 包含 root 自己
        int max_take = K * workers;
        std::vector<PT> batch;

        /* ① 批量弹 PT */
        for(int c=0; c<max_take && !priority.empty(); ++c){
            PT pt = priority.front(); priority.erase(priority.begin());
            /* 先把子 PT 插回队列 */
            for(PT &np : pt.NewPTs()){ CalProb(np); 
                /* 按概率插回 priority（与 PopNext 里同逻辑） */
                auto it = priority.begin();
                while(it!=priority.end() && it->prob >= np.prob) ++it;
                priority.insert(it,np);
            }
            batch.push_back(pt);
        }

        /* ② 打包 batch 为 sendBuf */
        std::vector<PackedPT> packed(batch.size());
        for(size_t i=0;i<batch.size();i++) PackPT(batch[i],packed[i]);

        int totalBytes = packed.size()*sizeof(PackedPT);
        MPI_Bcast(&totalBytes,1,MPI_INT,0,MPI_COMM_WORLD);   // 广播字节数
        /* ③ SCATTERV */
        int base = packed.size()/workers, extra = packed.size()%workers;
        std::vector<int> sendCounts(workers), displs(workers);
        int offset=0;
        for(int r=0;r<workers;r++){
            sendCounts[r] = (base + (r<extra)) * sizeof(PackedPT);
            displs[r] = offset;
            offset += sendCounts[r];
        }
        std::vector<uint8_t> sendBuf(reinterpret_cast<uint8_t*>(packed.data()),
                                     reinterpret_cast<uint8_t*>(packed.data())+totalBytes);

        int recvBytes = sendCounts[0];
        std::vector<uint8_t> recvBuf(recvBytes);
        MPI_Scatterv(sendBuf.data(),sendCounts.data(),displs.data(),MPI_BYTE,
                     recvBuf.data(),recvBytes,MPI_BYTE,
                     0,MPI_COMM_WORLD);

        /* ④ root 本地处理 */
        int localCnt = process_PT_buffer(recvBuf,m);

        /* ⑤ 汇总计数 */
        int globalCnt = 0;
        MPI_Reduce(&localCnt,&globalCnt,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

        total_guesses += globalCnt;                 // 更新全局计数
    }
    /*---------------- worker 逻辑 ----------------*/
    else {
        // int totalBytes;
        // MPI_Bcast(&totalBytes,1,MPI_INT,0,MPI_COMM_WORLD);
        // std::vector<uint8_t> recvBuf(totalBytes / size + 100);   // 粗估分配
        // /* root 会告诉每进程确切 bytes via Scatterv */
        // int recvBytes=0;           // runtime 得知
        // MPI_Scatterv(nullptr,nullptr,nullptr,MPI_BYTE,
        //              recvBuf.data(),recvBytes,MPI_BYTE,
        //              0,MPI_COMM_WORLD);

        // int localCnt = process_PT_buffer(recvBuf);
        // MPI_Reduce(&localCnt,nullptr,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    }
}