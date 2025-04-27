import tensorflow as tf
from PIL import Image
import nltk
from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer, TFVisionEncoderDecoderModel
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from third_party.keras_cv.stable_diffusion import StableDiffusion 
from third_party.keras_cv.diffusion_model import SpatialTransformer
from diffseg.utils import process_image, augmenter, vis_without_label, semantic_mask
from diffseg.segmentor import DiffSeg
import os

is_noun = lambda pos: pos[:2] == 'NN'
nltk.download('all')

# Inialize Stable Diffusion Model on GPU:0 
with tf.device('/GPU:0'):
  image_encoder = ImageEncoder()
  vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-1].output,
        )
  model = StableDiffusion(img_width=512, img_height=512)

# Initialize the new vision–language captioning model on GPU:1
with tf.device('/GPU:1'):
    image_processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    decoder_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_model = TFVisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", from_pt=True)
    
# The first time running this cell will be slow
# because the model needs to download and loads pre-trained weights.
img_root = "/home/angus/uce_images"
img_dirs = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
print("Found directories:", img_dirs)

for idx in img_dirs:
    image_path = os.path.join(img_root, idx, "img.jpg")
    print("Processing:", image_path)
    
    # Use the new captioning method instead of BLIP.
    with tf.device('/GPU:1'):
        img_for_caption = Image.open(image_path).convert("RGB")
        img_for_caption = img_for_caption.resize((224,224))
        pixel_values = image_processor(images=img_for_caption, return_tensors="tf").pixel_values
        generated_ids = caption_model.generate(pixel_values)
        prompt = decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        prompt += " with background"
        print("Caption:", prompt)

    with tf.device('/GPU:0'):
        img = process_image(image_path)
        img = augmenter(img)
        latent = vae(tf.expand_dims(img, axis=0), training=False)
        images, weight_64, weight_32, weight_16, weight_8, x_weights_64, x_weights_32, x_weights_16, x_weights_8 = model.text_to_image(
            prompt,
            batch_size=1,
            latent=latent,
            timestep=300
        )
        
    KL_THRESHOLD = [0.9]*3
    NUM_POINTS = 16
    REFINEMENT = True

    with tf.device('/GPU:0'):
        segmentor = DiffSeg(KL_THRESHOLD, REFINEMENT, NUM_POINTS)
        pred = segmentor.segment(weight_64, weight_32, weight_16, weight_8)
        # Always process the generated caption.
        tokenized = nltk.word_tokenize(prompt)
        nouns = [(i, word) for i, (word, pos) in enumerate(nltk.pos_tag(tokenized)) if is_noun(pos)]
        for i in range(len(images)):
            x_weight = segmentor.aggregate_x_weights(
                [x_weights_64[i], x_weights_32[i], x_weights_16[i], x_weights_8[i]],
                weight_ratio=[1.0, 1.0, 1.0, 1.0]
            )
            label_to_mask = segmentor.get_semantics(pred[i], x_weight[i], nouns, voting="majority")
            semantic_mask(images[i], pred[i], label_to_mask, save_individual=True, out_dir=f"./results/{idx}")
            vis_without_label(pred[i], images[i], save=True, dir=f"./results/{idx}", index=i, num_class=len(set(pred[i].flatten())))
            
    # Free intermediate GPU memory for this image
    import gc
    del img, latent, images, weight_64, weight_32, weight_16, weight_8, x_weights_64, x_weights_32, x_weights_16, x_weights_8
    gc.collect()

