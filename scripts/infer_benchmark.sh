#!/bin/bash

# Function to handle errors
error_handler() {
    echo "Error occurred in script at line: $1"
    exit 1
}

# Trap errors and call the error_handler function
trap 'error_handler $LINENO' ERR

# Iterate over the images in the directory
for i in {0..7}
do
    # Run the preprocess script with the specified arguments
    python inference_benchmark.py \
    --dataset_path datasets/$i \
    --model_path outputs/reproduce_output/$i \
    --output_path outputs/reproduce_gen_image \
    --seed 20
done