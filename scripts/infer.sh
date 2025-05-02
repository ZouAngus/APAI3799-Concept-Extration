MODEL_PATH="test/word_init_loss/00"
# make sure to change the model path

python inference.py \
    --model_path "outputs/${MODEL_PATH}" \
    --prompt "a photo of <asset1>" \
    --output_path "./inference/asset1.jpg" \