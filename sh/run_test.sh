DATASET=('wave1')            # 时间间隔
GPU_IDS=(0 1 2 3) # 可用的 GPU ID
# MODEL_SETTING=('wave') #数据集
MODEL_SETTING=('kernel_3_nogrid') #数据集

POSITION=('lu' 'ld' 'ru' 'rd' 'mid')

model_name='FNO'

# 计数器，用于分配 GPU
gpu_counter=0

# 创建日志目录（如果不存在）
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 网格搜索
for data in "${DATASET[@]}"; do
    for model in "${MODEL_SETTING[@]}"; do
        for p in "${POSITION[@]}"; do
            gpu_id=${GPU_IDS[$gpu_counter]}
            
            # 打印当前参数组合
            echo "Running kernel 3 with:"
            echo "data: $data, model: $model, position: $p"
            
            # 使用 nohup 在后台运行训练程序，并将输出重定向到日志文件
            nohup python test_position.py \
                --which_example $data \
                --which_model $model_name \
                --random_seed 1 \
                --which_device $gpu_id \
                --exp_name $model \
                --which_point $p > "${LOG_DIR}/test_${gpu_id}_data${dataset}_${model}_${p}.log" 2>&1 &
            
            # 更新 GPU 计数器
            gpu_counter=$(( (gpu_counter + 1) % ${#GPU_IDS[@]} ))
        done
    done
done

# 提示信息
echo "All training jobs have been started in the background."
echo "Logs are saved in the '${LOG_DIR}' directory."
echo "Use 'tail -f ${LOG_DIR}/*.log' to monitor the logs."