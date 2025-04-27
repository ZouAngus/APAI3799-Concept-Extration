import os
from pathlib import Path
from PIL import Image
from typing import Union, List

import torch
import clip

############################################################
# dataset_and_mask_directory = '../uce_images'
# generated_images_directory = './'
infer_img_path = "./evaluate/sim_identity"
############################################################

### helper function for loading the image
def load_semantic_mask(
    path: Union[str, Path],
    i: int,
    j: Union[str, int],
    first_only: bool = True
) -> Union[Image.Image, List[Image.Image]]:
    """
    Find and open the file(s) named "semantic_mask_{j}*" in the subdirectory
    infer_img_path/{i:02d}/.  Raises FileNotFoundError if nothing matches.
    """
    base = Path(path) / f"{i:02d}"
    pattern = f"semantic_mask_{j}*"
    matches = list(base.glob(pattern))
    
    if not matches:
        print(f"No files in {base!r} matching {pattern!r}")
        raise FileNotFoundError(f"No files in {base!r} matching {pattern!r}")
    
    images = [Image.open(p) for p in matches]
    return images[0] if first_only else images

clip_model, clip_preprocess = clip.load("ViT-B/16", device='cpu')

# we have the original images and the mask + the generated images corresponding to the mask, now calculate the similarity using CLIP
cosine_similarity = torch.nn.CosineSimilarity(dim=0)

similarity = 0.

for i in range(96):
    # --- REWRITTEN: collect & sort masks by their embedded index ---
    base_dir = os.path.join(infer_img_path, f"{i:02d}")
    mask_info = []
    for fname in os.listdir(base_dir):
        if not fname.startswith("semantic_mask_"):
            continue
        parts = fname.split("_")   # e.g. ["semantic","mask","3","dog.png"]
        try:
            idx = int(parts[2])
        except (IndexError, ValueError):
            continue
        mask_info.append((idx, fname))
    mask_info.sort(key=lambda x: x[0])
    # --------------------------------------------------------------

    similarity_j = 0.

    for idx, mask_file in mask_info:
        # load the mask
        mask = Image.open(os.path.join(base_dir, mask_file))

        # use the helper to handle multi-match if needed
        masked_image = load_semantic_mask(infer_img_path, i=i, j=idx)
        
        # get the list of generated images for the mask
        gen_images_path = os.path.join(base_dir, f"asset_{idx}")
        gen_images = [f for f in os.listdir(gen_images_path) if f.endswith(".jpg")]

        similarity_k = 0.
        for gen_image in gen_images:
            generated_image = Image.open(os.path.join(gen_images_path, gen_image))

            # Preprocess images
            masked_image_ = clip_preprocess(masked_image).unsqueeze(0)
            generated_image_ = clip_preprocess(generated_image).unsqueeze(0)

            # Get features
            with torch.no_grad():
                masked_features = clip_model.encode_image(masked_image_).float()
                masked_features /= masked_features.norm(dim=-1, keepdim=True)
                generated_features = clip_model.encode_image(generated_image_).float()
                generated_features /= generated_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            sim = cosine_similarity(masked_features[0], generated_features[0]).item()
            similarity_k += sim
        
        similarity_k /= len(gen_images)
        similarity_j += similarity_k

    # average over however many masks we actually found
    if mask_info:
        similarity_j /= len(mask_info)
    similarity += similarity_j

similarity /= 96

print(f"SIM Identity CLIP for: {infer_img_path}")
print(f"Similarity score: {similarity}")
