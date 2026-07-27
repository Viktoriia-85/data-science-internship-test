# Task 2: Sentinel-2 Seasonal Image Matching

Finds matching points between two Sentinel-2 satellite images of the same
place taken in different seasons (e.g. a snowy December scene and a green
June one), despite how different they look.

## Dataset

**[viktoriia85/sentinel2-seasonal-patches](https://www.kaggle.com/datasets/viktoriia85/sentinel2-seasonal-patches)**
(Kaggle) — 1876 patch pairs (512×512 px, 10 m resolution) cut from 10
Sentinel-2 L1C scenes over tile T36UYA (Chernihiv region, Ukraine), covering
2016 and 2019. Five seasonal pairs were selected to span different levels of
difficulty, from opposite seasons (snow vs. green vegetation) to the same
season three years apart. Full details, including how scenes and pairs were
chosen and how patch quality was checked, are in `dataset_creation.ipynb`.

Patches store all 13 raw Sentinel-2 bands plus a ready-made RGB preview
(TCI), at native resolution and unnormalized — no preprocessing decision
(band choice, normalization, cloud threshold) is baked into the data itself.
Total size: 8.6 GB.

**Known limitations:** narrow geographic coverage (one tile, no mountains or
coastline); approximate cloud statistics (small/thin clouds can be missed);
only one pair represents the strongest seasonal contrast (winter vs.
summer); Level-1C only (no atmospheric correction). See
`dataset_creation.ipynb` for details.

## Approach

`channel_matcher_comparison.ipynb` compares 5 matchers (SIFT, XFeat+LighterGlue,
ALIKED+LightGlue, Efficient-LoFTR, and RoMa) across 7 channels, using
pretrained weights only.

- **Default: ALIKED + LightGlue on channel B03** — best precision among
  sparse/semi-dense matchers tested (mma_3px = 0.85), lightweight, and
  produces explicit keypoints.
- **Optional: RoMa** — heavier (dense, DINOv2 backbone), but the only
  matcher tested with no seasonal accuracy drop at all. Available via
  `--matcher roma` for cases where robustness matters more than speed.

**Fine-tuning (`train.py`):** fine-tunes LightGlue's matcher on this dataset
with a frozen ALIKED detector, using a synthetic homography for ground
truth. The fine-tuned checkpoints did **not** improve on the pretrained
weights used by default, so `train.py` is kept as a documented experiment —
its checkpoints are not wired into `algorithm.py` / `inference.py`.

## Model weights

No custom-trained weights are shipped for this task. The default matcher
(ALIKED + LightGlue, and RoMa as the alternative) uses **pretrained**
weights only, downloaded automatically by `vismatch` on first use — no
manual download or link is needed to run `inference.py`.

The fine-tuned LightGlue checkpoints produced by `train.py` are not
published, since they did not outperform the pretrained weights (see
"Fine-tuning" above) and are not used by the inference pipeline.

## Setup

```bash
pip install -r requirements.txt
```

`dataset_creation.ipynb`, `channel_matcher_comparison.ipynb`, and
`demo.ipynb` were developed and run in Google Colab / Kaggle. If run
elsewhere, replace the Drive-mounting cells with local paths pointing to
this folder.

## Usage

**Run matching on a patch pair:**

```bash
python inference.py --patch-a scene_pair.npz --channel B03 \
    --matcher aliked-lightglue --output result.json --visualize result.png

# higher-quality, seasonally robust alternative
python inference.py --patch-a scene_pair.npz --channel B03 \
    --matcher roma --visualize result.png
```

`inference.py` also accepts raw per-band GeoTIFF/JP2 scenes instead of the
project's `.npz` format — see `--input-format geotiff` in the script's
`--help`.

**Fine-tune LightGlue** (experimental, see note above):

```bash
python train.py --dataset-path /path/to/kaggle/dataset --epochs 5 \
    --lr 1e-5 --channel B03 --checkpoint-dir ./checkpoints
```

See `demo.ipynb` for a walkthrough with visualized keypoints and matches on
three patch pairs of increasing difficulty.

## Potential improvements

See the project report (PDF) for proposed next steps, including a full
channel sweep for RoMa and testing on the excluded T36UXA tile as unseen
data.
