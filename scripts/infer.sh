MODEL_PATH="test/fined_tuned/02"
OUTPUT_DIR="inference/02"
# make sure to change the model path

python inference.py \
    --model_path "outputs/${MODEL_PATH}" \
    --prompt "a photo of <asset0> in a playground" \
    --output_path "$OUTPUT_DIR/asset0_playground.jpg" \

# python inference.py \
#     --model_path "outputs/${MODEL_PATH}" \
#     --prompt "a photo of <asset1>" \
#     --output_path "$OUTPUT_DIR/asset1.jpg" \

# python inference.py \
#     --model_path "outputs/${MODEL_PATH}" \
#     --prompt "a photo of <asset0> and <asset1>" \
#     --output_path ".$OUTPUT_DIR/asset0&1.jpg" \