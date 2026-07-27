"""
train.py — Fine-tune LightGlue's matcher on the Sentinel-2 seasonal patch
dataset, with a frozen detector (ALIKED by default).

Only LightGlue is trained, the detector stays frozen. Ground truth
correspondences are generated on the fly from a synthetic homography
applied to scene B of each patch pair.

Note: fine-tuned checkpoints from this script are not wired into
algorithm.py / inference.py. In testing, fine-tuning did not improve on
the pretrained aliked-lightglue results — the script is kept as a documented
experiment, not as part of the inference pipeline.

Usage:
    python train.py --dataset-path /path/to/kaggle/dataset --epochs 5 \\
        --lr 1e-5 --channel B03 --checkpoint-dir ./checkpoints

Requires:
    pip install lightglue @ git+https://github.com/cvg/LightGlue.git
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from lightglue import LightGlue, ALIKED
from lightglue.lightglue import normalize_keypoints
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from algorithm import build_channel


# Data selection and splitting
def get_clean_index(index, threshold):
    """Return only the patches where both scenes are below the given cloud %."""
    worst = index[["a_cloud_pct", "b_cloud_pct"]].max(axis=1)
    return index[worst < threshold].copy()


def split_train_val(index, cloud_threshold=5.0, n_splits=5, seed=0):
    """Group-aware train/val split by geographic block, so no block appears
    in both splits (avoids spatial leakage)."""
    pool = get_clean_index(index, cloud_threshold)
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = next(gkf.split(pool, groups=pool["block"]))
    train_set = pool.iloc[train_idx].reset_index(drop=True)
    val_set = pool.iloc[val_idx].reset_index(drop=True)
    return train_set, val_set


# Synthetic homography
def make_homography(size=512, rho=32, seed=0):
    """Random homography via 4-point corner perturbation (HomographyNet-style)."""
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    offsets = rng.uniform(-rho, rho, size=(4, 2)).astype(np.float32)
    dst = src + offsets
    return cv2.getPerspectiveTransform(src, dst)


def warp_patch(img, H, size=512):
    """Apply the homography, using reflected borders instead of black fill."""
    return cv2.warpPerspective(img, H, (size, size), borderMode=cv2.BORDER_REFLECT101)


# Dataset
class SeasonalMatchingDataset(Dataset):
    """One sample = one patch pair (scene A + warped scene B) on a single
    channel. A fresh homography is generated per sample (seeded by index,
    so still reproducible), giving exact pixel ground truth for training.
    """

    def __init__(self, patch_index, patches_dir, channel, rho=32, size=512):
        self.patch_index = patch_index.reset_index(drop=True)
        self.patches_dir = patches_dir
        self.channel = channel
        self.rho = rho
        self.size = size

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        row = self.patch_index.iloc[idx]
        data = np.load(os.path.join(self.patches_dir, row["file"]))

        H = make_homography(size=self.size, rho=self.rho, seed=idx)
        imgA = build_channel(data, "a", self.channel)
        imgB = warp_patch(build_channel(data, "b", self.channel), H, size=self.size)

        return {"imgA": imgA, "imgB": imgB, "H": H, "file": row["file"]}


def to_tensor(img, device):
    t = torch.from_numpy(img).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    return t.to(device)


# LightGlue internals: raw per-layer log-assignment
def lightglue_all_layer_scores(matcher, kpts0, kpts1, desc0, desc1, size0, size1):
    """Returns the log-assignment matrix from every LightGlue layer, using
    only its public submodules."""
    k0, k1 = normalize_keypoints(kpts0, size0), normalize_keypoints(kpts1, size1)
    d0, d1 = matcher.input_proj(desc0), matcher.input_proj(desc1)
    e0, e1 = matcher.posenc(k0), matcher.posenc(k1)
    all_scores = []
    for i in range(matcher.conf.n_layers):
        d0, d1 = matcher.transformers[i](d0, d1, e0, e1)
        scores, _ = matcher.log_assignment[i](d0, d1)
        all_scores.append(scores)
    return all_scores


def build_ground_truth(kptsA, kptsB, H, threshold=3.0):
    """Ground-truth correspondence from the known homography."""
    H_inv = np.linalg.inv(H)
    A, B = kptsA.cpu().numpy(), kptsB.cpu().numpy()
    A_h = np.concatenate([A, np.ones((len(A), 1))], axis=1)
    mapped = (H_inv @ A_h.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]

    dists = np.linalg.norm(mapped[:, None, :] - B[None, :, :], axis=2)
    nn_B, nn_dist = dists.argmin(axis=1), dists.min(axis=1)
    nn_A = dists.argmin(axis=0)

    gt0 = np.full(len(A), -1)
    for i in range(len(A)):
        j = nn_B[i]
        if nn_dist[i] <= threshold and nn_A[j] == i:
            gt0[i] = j
    gt1 = np.full(len(B), -1)
    for i, j in enumerate(gt0):
        if j >= 0:
            gt1[j] = i
    return (torch.from_numpy(gt0).long().to(kptsA.device),
            torch.from_numpy(gt1).long().to(kptsA.device))


def matching_loss(scores, gt0, gt1, unmatched_weight=0.5):
    """Negative log-likelihood for one layer's assignment matrix."""
    M, N = scores.shape[1] - 1, scores.shape[2] - 1
    matched = gt0 >= 0
    loss = 0.0
    if matched.any():
        idx_i = torch.where(matched)[0]
        loss = loss - scores[0, idx_i, gt0[matched]].mean()
    unmatched0 = gt0 == -1
    if unmatched0.any():
        loss = loss - unmatched_weight * scores[0, :M, N][unmatched0].mean()
    unmatched1 = gt1 == -1
    if unmatched1.any():
        loss = loss - unmatched_weight * scores[0, M, :N][unmatched1].mean()
    return loss


def matching_loss_weighted(all_scores, gt0, gt1, unmatched_weight=0.5):
    """Deep-supervision loss across all layers, weighted toward later
    layers."""
    n_layers = len(all_scores)
    weights = torch.linspace(0.2, 1.0, n_layers, device=all_scores[0].device)
    weights = weights / weights.sum()
    losses = torch.stack([matching_loss(s, gt0, gt1, unmatched_weight) for s in all_scores])
    return (losses * weights).sum()


# Training loop
def run_one_sample(matcher, extractor, sample, device, optimizer=None, training=True):
    imgA_t = to_tensor(sample["imgA"], device)
    imgB_t = to_tensor(sample["imgB"], device)
    H = sample["H"]

    with torch.no_grad():
        featsA = extractor.extract(imgA_t)
        featsB = extractor.extract(imgB_t)

    all_scores = lightglue_all_layer_scores(
        matcher, featsA["keypoints"], featsB["keypoints"],
        featsA["descriptors"], featsB["descriptors"],
        featsA["image_size"], featsB["image_size"],
    )
    gt0, gt1 = build_ground_truth(featsA["keypoints"][0], featsB["keypoints"][0], H)
    loss = matching_loss_weighted(all_scores, gt0, gt1)

    if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def train(args):
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    index = pd.read_csv(os.path.join(args.dataset_path, "index.csv"))
    patches_dir = os.path.join(args.dataset_path, "patches")
    train_set, val_set = split_train_val(index, cloud_threshold=args.cloud_threshold, seed=args.seed)
    print(f"Train: {len(train_set)} patches, Val: {len(val_set)} patches")

    train_dataset = SeasonalMatchingDataset(train_set, patches_dir, channel=args.channel)
    val_dataset = SeasonalMatchingDataset(val_set, patches_dir, channel=args.channel)

    extractor = ALIKED(max_num_keypoints=args.max_keypoints).eval().to(device)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False

    matcher = LightGlue(features="aliked").to(device)
    matcher.train()
    optimizer = torch.optim.Adam(matcher.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        matcher.train()
        train_losses = []
        for idx in tqdm(range(len(train_dataset)), desc=f"Epoch {epoch}/{args.epochs} [train]"):
            loss_val = run_one_sample(matcher, extractor, train_dataset[idx], device,
                                       optimizer=optimizer, training=True)
            train_losses.append(loss_val)

        matcher.eval()
        val_losses = []
        for idx in tqdm(range(len(val_dataset)), desc=f"Epoch {epoch}/{args.epochs} [val]"):
            with torch.no_grad():
                loss_val = run_one_sample(matcher, extractor, val_dataset[idx], device, training=False)
            val_losses.append(loss_val)

        epoch_train_loss, epoch_val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        print(f"Epoch {epoch}: train_loss={epoch_train_loss:.3f}, val_loss={epoch_val_loss:.3f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": matcher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "args": vars(args),
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    return history


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", required=True,
                   help="Path to the downloaded Kaggle dataset root (containing index.csv and patches/)")
    p.add_argument("--channel", default="B03", choices=["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "TCI"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--cloud-threshold", type=float, default=5.0,
                   help="Max allowed cloud %% (worst of the two scenes) for a patch to be used in training")
    p.add_argument("--max-keypoints", type=int, default=2048)
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
