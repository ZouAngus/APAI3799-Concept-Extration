# Description: Calculate the similarity between the original image and the generated image using DINO
import os

import torch
from PIL import Image
from torchvision import transforms

# Assuming DINO is properly installed and the model can be loaded
# You might need to adjust this based on your DINO installation
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
# disable batch norm
dino_model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

############################################################
# dataset_and_mask_directory = '../uce_images'
# generated_images_directory = './'
infer_img_path = "./evaluate/sim_composition"
############################################################

# we have the original images and the mask + the generated images corresponding to the mask, now calculate the similarity using dino encoder
cosine_similarity = torch.nn.CosineSimilarity(dim=0)

similarity = 0.

for i in range(96):
    # print(f"SIM_C DINO Processing {i}...")
    
    # load the original image, img.jpg
    current_path = os.path.join(infer_img_path, f"{i:02d}")
    original_image = Image.open(os.path.join(current_path, "img.jpg"))

    # get the list of generated images for the mask
    gen_images_path = current_path

    #####
    gen_images = [f for f in os.listdir(gen_images_path) if f.endswith(".jpg")] # buat ICE
    # gen_images = [f for f in glob.glob(os.path.join(generated_images_folder, "*/*.png"))]
    ####

    similarity_k = 0.
    for gen_image in gen_images:
        ######
        generated_image = Image.open(os.path.join(gen_images_path, gen_image)) # buat ICE
        # generated_image = Image.open(gen_image)
        ######

        # Preprocess images
        masked_image_ = transform(original_image).unsqueeze(0)  #.to(DEVICE)
        generated_image = transform(generated_image).unsqueeze(0) #.to(DEVICE)

        # Get features
        with torch.no_grad():
            masked_features = dino_model(masked_image_).float()
            masked_features /= masked_features.norm(dim=-1, keepdim=True)
            generated_features = dino_model(generated_image).float()
            generated_features /= generated_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        sim = cosine_similarity(masked_features[0], generated_features[0]).item()
        similarity_k += sim
        
    similarity_k /= len(gen_images) 
    similarity += similarity_k


similarity /= 96

print(f"SIM Compositionality DINO for: {infer_img_path}")
print(f"Similarity score: {similarity}")