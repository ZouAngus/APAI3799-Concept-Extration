#!/bin/bash
# set -e  # exit on error

export TF_CPP_MIN_LOG_LEVEL=2
segment_dir="./segment/segment_results"
sim_i_dir="./evaluate/sim_identity" 
sim_c_dir="./evaluate/sim_composition"

mkdir -p $sim_i_dir
mkdir -p $sim_c_dir

# Iterate over the uce_image in the directory one by one
for i in $(seq -w 0 95)

do
    # Training the model:
    python train.py \
        --instance_data_dir $segment_dir/$i/ \
        --class_data_dir outputs/preservation_images/ \
        --phase1_train_steps 500 \
        --phase2_train_steps 0 \
        --output_dir outputs/reproduce_output/$i/ \
        --use_8bit_adam \
        --set_grads_to_none \
        --noise_offset 0.1 \
        --t_dist 0.5 \
        --lambda_attention 1e-5 \
        --seed 20 \
        --prior_loss_weight 0.0 \

    # read mask direc
    mask_dir="$segment_dir/$i"
    # Prepare the Identity Similarity data
    mkdir -p "$sim_i_dir/$i"
    if [ -d "$mask_dir" ]; then
        mask_files=($(ls "$mask_dir"/mask_*.png 2>/dev/null))
        for mask_file in "${mask_files[@]}"; do
            mask_num=$(basename "$mask_file" | sed -E 's/mask_([0-9]+)\.png/\1/')
            
            # put the mask and the concept inference in the same directory
            asset_dir="$sim_i_dir/$i/asset_$mask_num"
            mkdir -p "$asset_dir"
            cp "$mask_dir/semantic_mask/semantic_mask_$mask_num"* "$sim_i_dir/$i/"
            for j in {1..8}; do
                python inference.py \
                    --model_path "outputs/reproduce_output/$i/" \
                    --prompt "a photo of <asset$mask_num>" \
                    --output_path "$asset_dir/asset${mask_num}_${j}.jpg"
            done
        done
    else
        echo "No mask directory found at $mask_dir"
    fi

    # Prepare the Compositional Similarity data
    mkdir -p "$sim_c_dir/$i"
    cp "$mask_dir/img.jpg" "$sim_c_dir/$i/"

    if [ -d "$mask_dir" ]; then
        mask_count=$(ls "$mask_dir"/mask_*.png 2>/dev/null | wc -l)
        assets=""
        for (( idx=0; idx<mask_count; idx++ )); do
            if [ $idx -eq 0 ]; then
                assets="<asset$idx>"
            else
                assets="${assets} and <asset$idx>"
            fi
        done
        comp_prompt="a photo of ${assets}"
        python inference.py \
            --model_path "outputs/reproduce_output/$i/" \
            --prompt "$comp_prompt" \
            --output_path "$sim_c_dir/$i/${comp_prompt}.jpg"
    fi
done

