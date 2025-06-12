/*
MPI 的main
*/
#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
// #include "md5_SIMD.h"
#include <iomanip>
#include <mpi.h> 
using namespace std;
using namespace chrono;

extern std::atomic<long long> g_generate_us;   // 声明
// extern std::atomic<long long> g_merge_us;

int process_PT_buffer(std::vector<uint8_t>& buf,model& g_model);

void BroadcastModel(model& m);

void FetchModel(model& m);


// 编译指令如下
// mpic++ -std=c++17 main_mpi_advanced.cpp train.cpp guessing_mpi_advanced.cpp md5.cpp -o main -O2

// qsub qsub_mpi.sh
// qstat
// 将 num 换成你的 qsub 序号
// qdel num.master_ubss1

// bash test.sh 1 1 1

/* worker 常驻循环：等待 root 的广播任务，N==-1 时退出 */
void WorkerLoop() {
    static model g_model;
    static bool inited = false;
    if (!inited)
    {
        FetchModel(g_model);
        inited = true;
    }
    while (true) {
        int ptBytes;                       // root 先广播 本轮 PT 序列化总字节
        MPI_Bcast(&ptBytes,1,MPI_INT,0,MPI_COMM_WORLD);
        if(ptBytes == -1) break;           // 终止标志

        /* 1. 接收 PT 数组 (Scatterv) */
        int recvBytes = 0;                 // root via Scatterv 决定
        std::vector<uint8_t> buf(ptBytes); // over‐alloc root 会调
        MPI_Scatterv(nullptr,nullptr,nullptr,MPI_BYTE,
                     buf.data(),recvBytes,MPI_BYTE,
                     0,MPI_COMM_WORLD);

        /* 2. 解析并顺序生成口令 */
        int localCnt = process_PT_buffer(buf,g_model);       // 函数稍后定义

        /* 3. 只回传计数即可，也可改发明文 */
        MPI_Reduce(&localCnt,nullptr,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    }
}

int RootMain()
{

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    BroadcastModel(q.m);          // ⬅ root 广播模型
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results_mine.txt");
    while (!q.priority.empty())
    {
        // q.PopNext();
        q.BatchGenerateMPI(32);
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                cout << "Generate total time: "
                     << std::fixed << std::setprecision(6)
                     << (g_generate_us.load() / 1e6) << " seconds" << endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }

            // int total = q.guesses.size();
            // int groupCount = total / 4;  // 整组数

            // // 依次对整组的每 4 个口令进行并行处理
            // for (int i = 0; i < groupCount; i++)
            // {
            //     std::string batch[4];
            //     for (int j = 0; j < 4; j++)
            //     {
            //         batch[j] = q.guesses[i * 4 + j];
            //     }
            //     // 定义输出的并行哈希结果：state_out[k][j] 表示第 j 个口令的第 k 个状态字
            //     bit32 state_parallel[4][4];
            //     MD5Hash_SIMD(batch, state_parallel);

            //     // 这里可以根据需要输出或记录每个口令的哈希结果
                
            //     // for (int j = 0; j < 4; j++) {
            //     //     a << batch[j] << "\t";
            //     //     for (int k = 0; k < 4; k++) {
            //     //         a << std::setw(8) << std::setfill('0') << hex << state_parallel[k][j] << " ";
            //     //     }
            //     //     a << endl;
            //     // }
                
            // }
            // // 如果剩余不足 4 个口令，单独处理
            // int remaining = total % 4;
            // if (remaining > 0)
            // {
            //     std::string batch[4];
            //     for (int i = 0; i < remaining; i++)
            //     {
            //         batch[i] = q.guesses[groupCount * 4 + i];
            //     }
            //     for (int i = remaining; i < 4; i++)
            //     {
            //         batch[i] = ""; // 补充空字符串，保证 4 个口令
            //     }
            //     bit32 state_parallel[4][4];
            //     MD5Hash_SIMD(batch, state_parallel);
                
            // }

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    return 0;

}


int main(int argc,char* argv[])
{
    MPI_Init(&argc,&argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank == 0) {
        /* root 执行原先所有串行逻辑（训练/优先队列 while 循环/MD5 计时） */
        RootMain();          
        /* 广播 “终止信号” 给 worker */
        int sentinel = -1;
        MPI_Bcast(&sentinel,1,MPI_INT,0,MPI_COMM_WORLD);
    }
    else {
        WorkerLoop();         
    }

    MPI_Finalize();
    return 0;
}