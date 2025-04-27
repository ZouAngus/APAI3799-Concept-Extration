import os
import argparse
import tensorflow as tf
from PIL import Image
import nltk
from transformers import AutoProcessor, TFBlipForConditionalGeneration
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from third_party.keras_cv.stable_diffusion import StableDiffusion 
from third_party.keras_cv.diffusion_model import SpatialTransformer
from diffseg.utils import process_image, augmenter, vis_without_label, semantic_mask
from diffseg.segmentor import DiffSeg

# Argument parsing for img_root and result_directory
parser = argparse.ArgumentParser(description="Apply DiffSeg on images")
parser.add_argument("--img_root", type=str, default="/home/angus/uce_images", help="Root directory for images")
parser.add_argument("--result_directory", type=str, default="./results", help="Directory to save results")
args = parser.parse_args()

is_noun = lambda pos: pos[:2] == 'NN'
nltk.download('all')

img_root = args.img_root
img_dirs = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
print("Found directories:", img_dirs)

# Initialize Stable Diffusion Model on GPU:2 
with tf.device('/GPU:2'):
    image_encoder = ImageEncoder()
    vae = tf.keras.Model(
        image_encoder.input,
        image_encoder.layers[-1].output,
    )
    model = StableDiffusion(img_width=512, img_height=512)
blip = None

# Optionally initialize a BLIP captioning model on GPU:3
with tf.device('/GPU:3'):
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
for idx in img_dirs:
    image_path = os.path.join(img_root, idx, "img.jpg")
    print("Processing:", image_path)
    
    if blip is not None:
        with tf.device('/GPU:3'):
            inputs = processor(images=Image.open(image_path), return_tensors="tf")
            out = blip.generate(**inputs)
            prompt = processor.decode(out[0], skip_special_tokens=True)
            ### separate background
            prompt += " with background"
            print("Caption:", prompt)
    else:
        prompt = None

    with tf.device('/GPU:2'):
        img = process_image(image_path)
        img = augmenter(img)
        latent = vae(tf.expand_dims(img, axis=0), training=False)
        images, weight_64, weight_32, weight_16, weight_8, x_weights_64, x_weights_32, x_weights_16, x_weights_8 = model.text_to_image(
            prompt,
            batch_size=1,
            latent=latent,
            timestep=300
        )
        
    KL_THRESHOLD = [1]*3
    NUM_POINTS = 16
    REFINEMENT = True

    with tf.device('/GPU:2'):
        segmentor = DiffSeg(KL_THRESHOLD, REFINEMENT, NUM_POINTS)
        pred = segmentor.segment(weight_64, weight_32, weight_16, weight_8)
        if blip is not None:
            tokenized = nltk.word_tokenize(prompt)
            nouns = [(i, word) for i, (word, pos) in enumerate(nltk.pos_tag(tokenized)) if is_noun(pos)]
        for i in range(len(images)):
            if blip is not None:
                x_weight = segmentor.aggregate_x_weights(
                    [x_weights_64[i], x_weights_32[i], x_weights_16[i], x_weights_8[i]],
                    weight_ratio=[1.0, 1.0, 1.0, 1.0]
                )
                label_to_mask = segmentor.get_semantics(pred[i], x_weight[i], nouns, voting="mean")
                semantic_mask(images[i], pred[i],  label_to_mask, save_individual=True , out_dir=f"{args.result_directory}/{idx}")
            # Use the passed parameter for the result directory
            vis_without_label(pred[i], images[i], save=True, dir=f"{args.result_directory}/{idx}", index=i, num_class=len(set(pred[i].flatten())))
            
    # Free intermediate GPU memory for this image
    import gc
    del img, latent, images, weight_64, weight_32, weight_16, weight_8, x_weights_64, x_weights_32, x_weights_16, x_weights_8
    gc.collect()

