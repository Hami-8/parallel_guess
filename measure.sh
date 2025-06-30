#!/usr/bin/env bash
set -euo pipefail

rm -f gpu.log cpu.log

APP=./main                 # 可执行文件
STDOUT=test.o              # 标准输出文件
STDERR=test.e              # 标准错误文件
INTERVAL=1
GPU_LOG=gpu.log
CPU_LOG=cpu.log

echo "[INFO] Launching ${APP}  (stdout→${STDOUT}, stderr→${STDERR})"

# -------- 在后台运行并重定向 --------
${APP} > "${STDOUT}" 2> "${STDERR}" &
APP_PID=$!
echo "[INFO] PID=${APP_PID}"

# ---------- GPU 监控 ----------
nvidia-smi dmon -s pucm -d ${INTERVAL} -o DT -f ${GPU_LOG} &
MON_GPU_PID=$!

# ---------- CPU 监控 ----------
pidstat -p ${APP_PID} ${INTERVAL} > ${CPU_LOG} &
MON_CPU_PID=$!

# ---------- 等待目标程序结束 ----------
wait ${APP_PID}

# ---------- 停止监控 ----------
kill ${MON_GPU_PID} 2>/dev/null || true
kill ${MON_CPU_PID} 2>/dev/null || true
sleep 1

# ---------- 汇总 ----------
echo "----- GPU summary -----"
awk 'NR>2 { sm+=$4; pcie+=$9+$10; n++ }
     END { printf("GPU sm avg     : %.1f %%\n", sm/n);
           printf("PCIe throughput: %.1f MB/s\n", pcie/n); }' ${GPU_LOG}

echo "----- CPU summary -----"
awk '/^[0-9]/ && $1!="Linux" { cpu+=$8; n++ }
     END { printf("CPU util avg   : %.1f %%\n", cpu/n); }' ${CPU_LOG}

echo "[INFO] Logs saved: ${GPU_LOG}, ${CPU_LOG}, ${STDOUT}, ${STDERR}"