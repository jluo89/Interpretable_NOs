DATASET=('wave1')            # Dataset
GPU_IDS=(0 1 2 3) # AVAILABLE GPU ID
MODEL_SETTING=('kernel_3_nogrid') # WHICH MODEL CONFIG

POSITION=('lu' 'ld' 'ru' 'rd' 'mid')

model_name='FNO'

gpu_counter=0

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

for data in "${DATASET[@]}"; do
    for model in "${MODEL_SETTING[@]}"; do
        for p in "${POSITION[@]}"; do
            gpu_id=${GPU_IDS[$gpu_counter]}
            
            echo "Running kernel 3 with:"
            echo "data: $data, model: $model, position: $p"
            
            nohup python test_position.py \
                --which_example $data \
                --which_model $model_name \
                --random_seed 1 \
                --which_device $gpu_id \
                --exp_name $model \
                --which_point $p > "${LOG_DIR}/test_${gpu_id}_data${dataset}_${model}_${p}.log" 2>&1 &
            
            gpu_counter=$(( (gpu_counter + 1) % ${#GPU_IDS[@]} ))
        done
    done
done

echo "All training jobs have been started in the background."
echo "Logs are saved in the '${LOG_DIR}' directory."
echo "Use 'tail -f ${LOG_DIR}/*.log' to monitor the logs."
