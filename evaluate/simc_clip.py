import os

import torch
import clip
from PIL import Image

# clip_model, clip_preprocess = clip.load("ViT-B/16", device="cuda" if torch.cuda.is_available() else "cpu")
clip_model, clip_preprocess = clip.load("ViT-B/16", device="cpu")


############################################################
# dataset_and_mask_directory = '../uce_images'
# generated_images_directory = './'
infer_img_path = "./evaluate/sim_composition"
############################################################

# we have the original images and the mask + the generated images corresponding to the mask, now calculate the similairyt using CLIP encoder
cosine_similarity = torch.nn.CosineSimilarity(dim=0)

similarity = 0.

for i in range(96):
    # print(f"SIM_C Clip Processing {i}...")
    
    # load the original image, img.jpg
    current_path = os.path.join(infer_img_path, f"{i:02d}")
    original_image = Image.open(os.path.join(current_path, "img.jpg"))

    # get the list of generated images for the mask
    gen_images_path = current_path

    #####
    gen_images = [f for f in os.listdir(gen_images_path) if f.endswith(">.jpg")] # buat ICE
    # gen_images = glob.glob(os.path.join(gen_images_path, "*/*.png"))
    # gen_images = [f for f in glob.glob(os.path.join(generated_images_folder, "*/*.png"))]
    ####


    similarity_k = 0.
    for gen_image in gen_images:

        ######
        generated_image = Image.open(os.path.join(gen_images_path, gen_image)) # buat ICE
        # generated_image = Image.open(gen_image)
        ######

        # Preprocess images
        masked_image_ = clip_preprocess(original_image).unsqueeze(0)  #.to(DEVICE)
        generated_image = clip_preprocess(generated_image).unsqueeze(0) #.to(DEVICE)

        # Get features
        with torch.no_grad():
            masked_features = clip_model.encode_image(masked_image_).float()
            # masked_features /= masked_features.norm(dim=-1, keepdim=True)
            generated_features = clip_model.encode_image(generated_image).float()
            # generated_features /= generated_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        sim = cosine_similarity(masked_features[0], generated_features[0]).item()
        similarity_k += sim
        
    similarity_k /= len(gen_images) 
    similarity += similarity_k


similarity /= 96

print(f"SIM Compositionality CLIP for: {infer_img_path}")
print(f"Similarity score: {similarity}")