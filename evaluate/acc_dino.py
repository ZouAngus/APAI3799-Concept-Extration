import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
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

def load_dino(device="cuda"):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.requires_grad_(False)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        ),
    ])
    return model.to(device), transform

@torch.no_grad()
def extract_feats(
    model: torch.nn.Module,
    transform,
    images: List[Image.Image],
    device="cuda"
) -> torch.Tensor:
    feats = []
    for img in images:
        x = transform(img).unsqueeze(0).to(device)
        out = model(x)
        # DINO returns a tuple (feat, aux) on ResNet50; pick first
        f = out[0] if isinstance(out, (tuple,list)) else out
        f = F.normalize(f, dim=-1)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)  # [N, D]

def build_mask_prototypes(
    root: Path,
    model, transform,
    device="cuda"
) -> Dict[str, torch.Tensor]:
    protos: Dict[str, torch.Tensor] = {}
    for concept_dir in sorted(root.iterdir()):
        if not concept_dir.is_dir(): continue
        i = int(concept_dir.name)
        for mask_fp in sorted(concept_dir.glob("semantic_mask_*")):
            raw = mask_fp.name[len("semantic_mask_"):]  # "0_extra.png"
            raw = raw.split(".")[0]                     # "0_extra"
            j   = raw.split("_")[0]                     # "0"
            mask_img = load_semantic_mask(root, i, j, first_only=True)
            feat = extract_feats(model, transform, [mask_img], device).squeeze(0)
            feat = feat / feat.norm()
            key = f"{i:02d}__semantic_mask_{j}"
            protos[key] = feat
    return protos

def load_query_feats_labels(
    root: Path,
    model, transform,
    device="cuda"
):
    feats_list: List[torch.Tensor] = []
    labels_list: List[str]         = []
    for concept_dir in sorted(root.iterdir()):
        if not concept_dir.is_dir(): continue
        i = int(concept_dir.name)
        for asset_dir in sorted(concept_dir.iterdir()):
            if not asset_dir.is_dir() or not asset_dir.name.startswith("asset"):
                continue
            imgs = [
                Image.open(p).convert("RGB")
                for p in sorted(asset_dir.glob("*.jpg"))
            ]
            batch = extract_feats(model, transform, imgs, device)  # [8, D]
            avg_f = batch.mean(dim=0, keepdim=True)                # [1, D]
            avg_f = F.normalize(avg_f, dim=-1)
            k = asset_dir.name.split("_",1)[1]
            label = f"{i:02d}__semantic_mask_{k}"
            feats_list.append(avg_f)
            labels_list.append(label)

    q_feats = torch.cat(feats_list, dim=0)  # [M, D]
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
    inds       = np.argsort(-sims, axis=1)                         # [M, P]
    M = len(query_labels)
    results: Dict[int, float] = {}
    for k in ks:
        correct = sum(
            1 for i, lbl in enumerate(query_labels)
            if lbl in [proto_keys[idx] for idx in inds[i, :k]]
        )
        results[k] = correct / M
    return results

if __name__ == "__main__":
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    sim_root = Path("evaluate") / "sim_identity"

    # print("Loading DINO…")
    model, transform = load_dino(device)

    # print("Building DINO mask-level prototypes…")
    prototypes = build_mask_prototypes(sim_root, model, transform, device)
    # print(f" → {len(prototypes)} prototypes built\n")

    # print("Loading & averaging query assets…")
    q_feats, q_labels = load_query_feats_labels(sim_root, model, transform, device)
    # print(f" → {q_feats.shape[0]} query samples loaded\n")

    acc = topk_accuracy(prototypes, q_feats, q_labels, ks=[1,3])
    print(f"DINO ACC@1 = {acc[1]*100:.2f}%   ACC@3 = {acc[3]*100:.2f}%")
