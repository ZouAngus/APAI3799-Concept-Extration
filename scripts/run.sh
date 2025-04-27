#!/bin/bash

# Function to handle errors
error_handler() {
    echo "Error occurred in script at line: $1"
    exit 1
}

# Trap errors and call the error_handler function
trap 'error_handler $LINENO' ERR

python train_vanilla.py \
    --instance_data_dir ./segment/results-1/00\
    --class_data_dir outputs/preservation_images/ \
    --phase1_train_steps 500 \
    --phase2_train_steps 0 \
    --output_dir outputs/reproduce_output/vanilla/0 \
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