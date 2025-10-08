#!/bin/bash

echo "正在当前目录下启动 6 个并行的后台进程..."
echo "--------------------------------------------------------"

# 循环 t 从 1 到 6
for t in {1..6}
do
    # 定义日志文件名，它将在当前目录下创建
    LOG_FILE="wave_generation_t_${t}.log"

    # 使用 nohup 在后台启动 Python 程序
    # 输出会被重定向到当前目录下的 LOG_FILE
    nohup python wave_generation.py --t $t > "$LOG_FILE" 2>&1 &

    # 打印信息，告诉用户哪个进程已经启动
    echo "已启动 t=$t 的进程，日志文件: ./$LOG_FILE"
done

echo "--------------------------------------------------------"
echo "所有 6 个进程都已在后台启动。"
echo "你可以安全地关闭此终端。"
echo "要检查它们的状态, 请使用: ps -ef | grep 'wave_generation.py'"