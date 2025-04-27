#!/usr/bin/env python
"""
evaluate/acc_clip.py

— Build one prototype per (numeric) mask index j
  by loading the first matching file with load_semantic_mask.

— Build one query feature per asset_{k} by averaging its 8 images.

— Compute ACC@1, ACC@3.
"""

import torch
import clip
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union
import numpy as np

def load_semantic_mask(
    path: Union[str, Path],
    i: int,
    j: Union[str, int],
    first_only: bool = True
) -> Image.Image:
    base    = Path(path) / f"{i:02d}"
    pattern = f"semantic_mask_{j}*"
    matches = list(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No mask in {base!r} matching {pattern!r}")
    return Image.open(matches[0]).convert("RGB")

def load_clip_model(device="cuda"):
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()
    return model, preprocess

@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    preprocess,
    images: List[Image.Image],
    device="cuda"
) -> torch.Tensor:
    feats = []
    for img in images:
        x = preprocess(img).unsqueeze(0).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)  # [N, D]

def build_mask_prototypes(
    sim_identity_root: Path,
    model, preprocess,
    device="cuda"
) -> Dict[str, torch.Tensor]:
    protos: Dict[str, torch.Tensor] = {}
    for concept_dir in sorted(sim_identity_root.iterdir()):
        if not concept_dir.is_dir(): 
            continue

        i = int(concept_dir.name)  # e.g. 0, 1, … 95
        # for each mask file in this folder
        for mask_fp in sorted(concept_dir.glob("semantic_mask_*")):
            # extract numeric j before any underscore/suffix
            raw = mask_fp.name[len("semantic_mask_"):]  # e.g. "0_extra.png"
            raw = raw.split(".")[0]                     # "0_extra"
            j   = raw.split("_")[0]                     # "0"

            # load exactly that j’s mask
            mask_img = load_semantic_mask(sim_identity_root, i, j, first_only=True)

            # extract its CLIP feature
            feat = extract_features(model, preprocess, [mask_img], device).squeeze(0)
            feat = feat / feat.norm()

            key = f"{i:02d}__semantic_mask_{j}"
            protos[key] = feat

    return protos

def load_query_features_and_labels(
    sim_identity_root: Path,
    model, preprocess,
    device="cuda"
):
    feats_list: List[torch.Tensor] = []
    labels_list: List[str]         = []

    for concept_dir in sorted(sim_identity_root.iterdir()):
        if not concept_dir.is_dir(): 
            continue

        i = int(concept_dir.name)
        for asset_dir in sorted(concept_dir.iterdir()):
            if not asset_dir.is_dir() or not asset_dir.name.startswith("asset"):
                continue

            # load & average the 8 generated images
            imgs = [
                Image.open(p).convert("RGB")
                for p in sorted(asset_dir.glob("*.jpg"))
            ]
            batch = extract_features(model, preprocess, imgs, device)  # [8, D]
            avg_f = batch.mean(dim=0, keepdim=True)                    # [1, D]
            avg_f = avg_f / avg_f.norm(dim=-1, keepdim=True)

            k     = asset_dir.name.split("_",1)[1]                     # numeric j
            label = f"{i:02d}__semantic_mask_{k}"

            feats_list.append(avg_f)
            labels_list.append(label)

    q_feats  = torch.cat(feats_list, dim=0)  # [M, D]
    return q_feats, labels_list

def topk_accuracy(
    prototypes: Dict[str, torch.Tensor],
    query_feats: torch.Tensor,
    query_labels: List[str],
    ks: List[int] = [1,3]
) -> Dict[int, float]:
    proto_keys = list(prototypes.keys())
    proto_mat  = torch.stack([prototypes[k] for k in proto_keys])  # [P, D]
    sims       = (query_feats @ proto_mat.T).numpy()               # [M, P]
    topinds    = np.argsort(-sims, axis=1)                         # [M, P]

    M = len(query_labels)
    results: Dict[int, float] = {}
    for k in ks:
        correct = 0
        for i, lbl in enumerate(query_labels):
            topk = topinds[i, :k]
            preds = [proto_keys[idx] for idx in topk]
            if lbl in preds:
                correct += 1
        results[k] = correct / M

    return results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim_root = Path("evaluate") / "sim_identity"

    # print("Loading CLIP…")
    model, preprocess = load_clip_model(device)

    # print("Building normalized mask‐level prototypes…")
    prototypes = build_mask_prototypes(sim_root, model, preprocess, device)
    print(f" → {len(prototypes)} prototypes built\n")

    # print("Loading & averaging query assets…")
    q_feats, q_labels = load_query_features_and_labels(sim_root, model, preprocess, device)
    print(f" → {q_feats.shape[0]} query samples loaded\n")

    acc = topk_accuracy(prototypes, q_feats, q_labels, ks=[1,3])
    print(f"ACC@1 = {acc[1]*100:.2f}%   ACC@3 = {acc[3]*100:.2f}%")
