"""
algorithm.py — Sentinel-2 seasonal image matching pipeline.

This module builds a chosen input channel from a raw Sentinel-2 patch and
runs a chosen matcher on a pair of scenes (e.g. two different seasons of
the same location). It doesn't train anything - all matchers are used
with their pretrained weights.

Supported channels:    B02, B03, B04, B08, B11, NDVI, NDWI, TCI

Supported matchers:
    aliked-lightglue  — best precision among sparse/semi-dense matchers
                         tested (mma_3px ~0.85 on B03); default choice.
    roma               — heaviest model (DINOv2 backbone), but the only
                         matcher tested that showed no seasonal degradation
                         at all. Recommended when quality matters more
                         than speed/memory.
    xfeat-lightglue    — fastest deep matcher tested (~3s/pair on CPU).
    sift-lightglue     — SIFT keypoints, LightGlue matching.
    eloftr             — semi-dense (Efficient-LoFTR), no explicit keypoints.
    sift               — classical baseline, no learned weights, useful as
                         a fast sanity check.

Usage:
    from algorithm import match_images
    result = match_images(patch_npz_a, patch_npz_b, channel="B03",
                           matcher_name="aliked-lightglue")
"""

import glob
import os
import tempfile

import cv2
import numpy as np

SUPPORTED_CHANNELS = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "TCI"]
SUPPORTED_MATCHERS = [
    "aliked-lightglue", "roma", "xfeat-lightglue", "sift-lightglue", "eloftr", "sift",
]

_VISMATCH_NAME = {
    "aliked-lightglue": "aliked-lightglue",
    "roma": "roma",
    "xfeat-lightglue": "xfeat-lightglue",
    "sift-lightglue": "sift-lightglue",
    "eloftr": "eloftr",
}

# Cache loaded matchers so repeated calls don't reload weights from disk/network every time.
_matcher_cache = {}


# Channel construction

def _stretch_to_uint8(a, lo=2, hi=98):
    """Percentile stretch to 0-255, the same normalization used for every
    channel candidate throughout the project - keeps channels comparable."""
    a = a.astype(np.float32)
    p_lo, p_hi = np.percentile(a, [lo, hi])
    scaled = np.clip((a - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def build_channel(npz_data, prefix, channel):
    """Build a single channel (grayscale, uint8, 512x512) for one scene
    ('a' or 'b') of a patch.

    npz_data: dict-like object exposing '{prefix}_{band}' keys — either a
              loaded .npz archive or a plain dict built by load_scene_from_geotiffs()
              below, for matching raw, freshly downloaded scenes.
    prefix:   "a" or "b" — which scene of the pair.
    channel:  one of SUPPORTED_CHANNELS.
    """
    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(f"Unknown channel '{channel}'. Supported: {SUPPORTED_CHANNELS}")

    if channel == "TCI":
        return cv2.cvtColor(npz_data[f"{prefix}_TCI"], cv2.COLOR_RGB2GRAY)

    if channel in ("B02", "B03", "B04", "B08"):
        raw = npz_data[f"{prefix}_{channel}"].astype(np.float32)
        return _stretch_to_uint8(raw)

    if channel == "B11":
        raw = npz_data[f"{prefix}_B11"].astype(np.float32)
        if raw.shape != (512, 512):
            raw = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_LINEAR)
        return _stretch_to_uint8(raw)

    if channel == "NDVI":
        b04 = npz_data[f"{prefix}_B04"].astype(np.float32)
        b08 = npz_data[f"{prefix}_B08"].astype(np.float32)
        ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
        return _stretch_to_uint8(ndvi)

    if channel == "NDWI":
        b03 = npz_data[f"{prefix}_B03"].astype(np.float32)
        b08 = npz_data[f"{prefix}_B08"].astype(np.float32)
        ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
        return _stretch_to_uint8(ndwi)

    raise AssertionError("unreachable")  # all SUPPORTED_CHANNELS handled above


# Raw GeoTIFF / JP2 input support (for matching freshly downloaded scenes
# that were never packaged into the project's .npz format)

_BANDS_FOR_CHANNEL = {
    "B02": ["B02"], "B03": ["B03"], "B04": ["B04"], "B08": ["B08"], "B11": ["B11"],
    "NDVI": ["B04", "B08"], "NDWI": ["B03", "B08"], "TCI": ["TCI"],
}


def _read_band_file(path, target_size=512):
    """Read one band from a GeoTIFF/JP2 file and resize to the target grid.
    Uses rasterio if available (handles georeferenced formats properly);
    falls back to a plain image read for ordinary TIFFs.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
    except ImportError:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if arr.shape != (target_size, target_size):
        arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return arr


def _find_band_file(band_dir, band_code):
    """Locate the file for a given band inside a directory, matching
    standard Sentinel-2 naming conventions (e.g. '..._B03_10m.jp2',
    'B03.tif') — takes the first match if several are found.
    """
    patterns = [f"*{band_code}*.jp2", f"*{band_code}*.tif", f"*{band_code}*.tiff"]
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(band_dir, pattern)))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file found for band '{band_code}' in {band_dir}")


def load_scene_from_geotiffs(band_dir, channel, target_size=512):
    """Load only the raw bands needed for `channel` from a directory of
    per-band GeoTIFF/JP2 files.

    band_dir: directory containing one file per band, named so the band
              code appears in the filename (standard Sentinel-2 SAFE
              naming works, e.g. 'T36UYA_..._B03_10m.jp2'; simple names
              like 'B03.tif' work too).
    channel:  one of SUPPORTED_CHANNELS — determines which band file(s)
              are actually read.

    Returns a plain dict with unprefixed band keys (e.g. {'B03': array}),
    ready to be combined into the prefixed format build_channel() expects
    via combine_scenes().
    """
    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(f"Unknown channel '{channel}'. Supported: {SUPPORTED_CHANNELS}")

    bands = {}
    for band_code in _BANDS_FOR_CHANNEL[channel]:
        if band_code == "TCI":
            path = _find_band_file(band_dir, "TCI")
            try:
                import rasterio
                with rasterio.open(path) as src:
                    rgb = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
            except ImportError:
                rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (target_size, target_size):
                rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            bands["TCI"] = rgb
        else:
            path = _find_band_file(band_dir, band_code)
            bands[band_code] = _read_band_file(path, target_size)
    return bands


def combine_scenes(bands_a, bands_b):
    """Combine two unprefixed band dicts (from load_scene_from_geotiffs)
    into the single '{prefix}_{band}'-keyed dict that build_channel() and
    match_images() expect — the same shape as a loaded .npz patch.
    """
    combined = {}
    combined.update({f"a_{band}": arr for band, arr in bands_a.items()})
    combined.update({f"b_{band}": arr for band, arr in bands_b.items()})
    return combined


# Matcher loading
def _resolve_device(device):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _get_vismatch_matcher(name, device):
    key = (name, device)
    if key not in _matcher_cache:
        from vismatch import get_matcher
        _matcher_cache[key] = get_matcher(_VISMATCH_NAME[name], device=device)
    return _matcher_cache[key]


# Matching
def _match_with_sift(imgA, imgB, ratio_thresh=0.75):
    """Classical baseline: SIFT keypoints + brute-force ratio-test matching.
    No learned weights, no vismatch dependency
    """
    sift = cv2.SIFT_create()
    kpA, desA = sift.detectAndCompute(imgA, None)
    kpB, desB = sift.detectAndCompute(imgB, None)

    if desA is None or desB is None or len(desA) < 2 or len(desB) < 2:
        return _empty_result(kpA, kpB)

    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(desA, desB, k=2)
    good = [m for m, n in raw_matches if m.distance < ratio_thresh * n.distance]

    matched_a = np.array([kpA[m.queryIdx].pt for m in good], dtype=np.float32)
    matched_b = np.array([kpB[m.trainIdx].pt for m in good], dtype=np.float32)

    return {
        "keypointsA": np.array([kp.pt for kp in kpA], dtype=np.float32),
        "keypointsB": np.array([kp.pt for kp in kpB], dtype=np.float32),
        "matched_kptsA": matched_a,
        "matched_kptsB": matched_b,
        "num_matches": len(good),
    }


def _empty_result(kpA=None, kpB=None):
    kpA = np.array([kp.pt for kp in kpA], dtype=np.float32) if kpA else np.zeros((0, 2), dtype=np.float32)
    kpB = np.array([kp.pt for kp in kpB], dtype=np.float32) if kpB else np.zeros((0, 2), dtype=np.float32)
    return {
        "keypointsA": kpA, "keypointsB": kpB,
        "matched_kptsA": np.zeros((0, 2), dtype=np.float32),
        "matched_kptsB": np.zeros((0, 2), dtype=np.float32),
        "num_matches": 0,
    }


def match_images(
    patch_data_a,
    patch_data_b,
    channel="B03",             # "B02" | "B03" | "B04" | "B08" | "B11" | "NDVI" | "NDWI" | "TCI"
    matcher_name="aliked-lightglue",  # "aliked-lightglue" | "roma" | "xfeat-lightglue" | "sift-lightglue" | "eloftr" | "sift"
    scene_a_prefix="a",
    scene_b_prefix="b",
    device="auto",
):
    """Match two scenes of a Sentinel-2 patch pair.

    Returns a dict with:
        keypointsA, keypointsB   — all detected keypoints, shape (N, 2)
                                    (empty arrays for matchers without an
                                    explicit detector, e.g. "roma", "eloftr")
        matched_kptsA, matched_kptsB — matched point pairs, shape (M, 2) each
        num_matches               — M, the number of matched pairs
    """
    if matcher_name not in SUPPORTED_MATCHERS:
        raise ValueError(f"Unknown matcher '{matcher_name}'. Supported: {SUPPORTED_MATCHERS}")

    device = _resolve_device(device)
    imgA = build_channel(patch_data_a, scene_a_prefix, channel)
    imgB = build_channel(patch_data_b, scene_b_prefix, channel)

    if matcher_name == "sift":
        return _match_with_sift(imgA, imgB)

    matcher = _get_vismatch_matcher(matcher_name, device)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_a = os.path.join(tmp_dir, "a.png")
        path_b = os.path.join(tmp_dir, "b.png")
        cv2.imwrite(path_a, imgA)
        cv2.imwrite(path_b, imgB)

        img0 = matcher.load_image(path_a, resize=512)
        img1 = matcher.load_image(path_b, resize=512)
        result = matcher(img0, img1)

    return {
        "keypointsA": result["all_kpts0"],
        "keypointsB": result["all_kpts1"],
        "matched_kptsA": result["matched_kpts0"],
        "matched_kptsB": result["matched_kpts1"],
        "num_matches": len(result["matched_kpts0"]),
    }
