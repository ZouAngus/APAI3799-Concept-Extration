#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

DATA_DIR="./segment/segment_results/00"
MODEL_PATH="./outputs/test/word_init_dynamic/00"


# Function to handle errors
error_handler() {
    echo "Error occurred in script at line: $1"
    exit 1
}

# Trap errors and call the error_handler function
trap 'error_handler $LINENO' ERR

mkdir -p $MODEL_PATH

python train_init_attn_ctl.py \
    --instance_data_dir $DATA_DIR \
    --class_data_dir outputs/preservation_images/ \
    --phase1_train_steps 500 \
    --phase2_train_steps 0 \
    --output_dir $MODEL_PATH \
    --use_8bit_adam \
    --set_grads_to_none \
    --noise_offset 0.1 \
    --t_dist 0.5 \
    --lambda_attention 1e-5 \
    --seed 20 \
    --prior_loss_weight 0.0 \

# # Iterate over the images in the directory
# for i in {0..7}
# do
#     # Run the preprocess script with the specified arguments
#     python train.py \
#         --instance_data_dir datasets/$i/ \
#         --class_data_dir outputs/preservation_images/ \
#         --phase1_train_steps 500 \
#         --phase2_train_steps 0 \
#         --output_dir outputs/reproduce_output/$i/ \
#         --use_8bit_adam \
#         --set_grads_to_none \
#         --noise_offset 0.1 \
#         --t_dist 0.5 \
#         --lambda_attention 1e-5 \
#         --seed 20 \
#         --prior_loss_weight 0.0 \

# done