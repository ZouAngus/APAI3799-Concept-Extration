MODEL_PATH="word_init/00"
# make sure to change the model path

python inference.py \
    --model_path "outputs/${MODEL_PATH}" \
    --prompt "a photo of <asset0>" \
    --output_path "./inference/asset0.jpg" \