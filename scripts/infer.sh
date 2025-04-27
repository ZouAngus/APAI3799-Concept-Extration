MODEL_PATH="/reproduce_output/vanilla/0"
# make sure to change the model path

python inference.py \
    --model_path "outputs/${MODEL_PATH}" \
    --prompt "a photo of <asset0>" \
    --output_path "asset0.jpg"