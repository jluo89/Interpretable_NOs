#!/bin/bash

# Define the parameter sets to loop through
# DATASET=('wave1' 'wave2' 'wave3' 'wave4' 'wave5' 'wave6') # Datasets (representing different time intervals)
# DATASET=('wave1' 'wave2') # Datasets (representing different time intervals)
DATASET=('wave1') # Datasets (representing different time intervals)

GPU_IDS=(0 1 2 3)                                         # Available GPU IDs
# MODEL_SETTING=('wave_basis_512' 'wave_basis_256')       # Example of other model settings
# MODEL_SETTING=('FNOL' 'FNO')                            # Example of other model settings
MODEL_SETTING=('FNO')                                     # Model settings to use
EXP_NAMES=('kernel_3_nogrid')                             # Experiment name configuration

# Counter for GPU assignment
gpu_counter=0

# Create log directory if it doesn't exist
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Loop through all parameter combinations
for data in "${DATASET[@]}"; do
    for model in "${MODEL_SETTING[@]}"; do
        for exp_name in "${EXP_NAMES[@]}"; do
            # Assign a GPU from the list in a round-robin fashion
            gpu_id=${GPU_IDS[$gpu_counter]}
            
            # Print the current parameter combination being run
            echo "Running kernel ${model} with:"
            echo "data: $data, model: $model, exp_name: $exp_name on GPU: $gpu_id"
            
            # Run the training script in the background using nohup and redirect output to a log file
            nohup python train.py \
                --which_example "$data" \
                --which_model "$model" \
                --random_seed 1 \
                --which_device "$gpu_id" \
                --exp_name "$exp_name" \
                --which_point mid > "${LOG_DIR}/data_${data}_model_${model}_exp_${exp_name}_gpu_${gpu_id}.log" 2>&1 &
            
            # Update the GPU counter
            gpu_counter=$(( (gpu_counter + 1) % ${#GPU_IDS[@]} ))
        done
    done
done

# Informational messages
echo "All training jobs have been started in the background."
echo "Logs are saved in the '${LOG_DIR}' directory."
echo "Use 'tail -f ${LOG_DIR}/*.log' to monitor the logs."