"""
inference.py — Apply the matching pipeline (algorithm.py) to a pair of
Sentinel-2 scenes and save the result.

This script doesn't evaluate correctness (there is no ground truth for a
genuinely new pair of images) - it only runs detection + matching and
reports/saves what was found.

Usage:
    python inference.py --patch-a scene_pair.npz --patch-b scene_pair.npz \\
        --channel B03 --matcher aliked-lightglue \\
        --output result.json --visualize result.png

    # scene_pair.npz is the usual dataset format, holding both scenes under
    # "a_*"/"b_*" prefixed keys. --patch-a and --patch-b can also point to
    # two DIFFERENT .npz files if you want to match scenes from unrelated
    # patches
"""

import argparse
import json
import time

import numpy as np

from algorithm import (SUPPORTED_CHANNELS, SUPPORTED_MATCHERS, build_channel,
                        combine_scenes, load_scene_from_geotiffs, match_images)


def load_patch(path):
    return np.load(path)


def load_input(args):
    """Return (patch_data_a, patch_data_b) ready for match_images(), built
    either from an existing .npz patch or from raw per-band GeoTIFF/JP2
    directories, depending on --input-format.
    """
    if args.input_format == "npz":
        patch_a = load_patch(args.patch_a)
        patch_b = load_patch(args.patch_b) if args.patch_b != args.patch_a else patch_a
        return patch_a, patch_b

    # geotiff: read only the bands the chosen channel actually needs
    bands_a = load_scene_from_geotiffs(args.scene_a_dir, args.channel)
    bands_b = load_scene_from_geotiffs(args.scene_b_dir, args.channel)
    combined = combine_scenes(bands_a, bands_b)
    return combined, combined


def save_result(result, output_path):
    """Save keypoints/matches to JSON (small, human-readable) or .npz,
    based on the output file extension."""
    if output_path.endswith(".npz"):
        np.savez(output_path,
                 keypointsA=result["keypointsA"], keypointsB=result["keypointsB"],
                 matched_kptsA=result["matched_kptsA"], matched_kptsB=result["matched_kptsB"])
        return

    serializable = {
        "num_matches": result["num_matches"],
        "keypointsA": np.asarray(result["keypointsA"]).tolist(),
        "keypointsB": np.asarray(result["keypointsB"]).tolist(),
        "matched_kptsA": np.asarray(result["matched_kptsA"]).tolist(),
        "matched_kptsB": np.asarray(result["matched_kptsB"]).tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f)


def visualize(imgA, imgB, result, output_path):
    import matplotlib.pyplot as plt

    h, w = imgA.shape
    fig, ax = plt.subplots(figsize=(12, 6))
    canvas = np.concatenate([imgA, imgB], axis=1)
    ax.imshow(canvas, cmap="gray")

    matched_a = np.asarray(result["matched_kptsA"])
    matched_b = np.asarray(result["matched_kptsB"])
    for ptA, ptB in zip(matched_a, matched_b):
        ax.plot([ptA[0], ptB[0] + w], [ptA[1], ptB[1]],
                color="lime", linewidth=0.4, alpha=0.6)
        ax.plot(ptA[0], ptA[1], "o", color="cyan", markersize=1.5)
        ax.plot(ptB[0] + w, ptB[1], "o", color="cyan", markersize=1.5)

    ax.set_title(f"{result['num_matches']} matches")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args):
    patch_a, patch_b = load_input(args)

    start = time.time()
    result = match_images(
        patch_a, patch_b,
        channel=args.channel,
        matcher_name=args.matcher,
        scene_a_prefix=args.scene_a_prefix,
        scene_b_prefix=args.scene_b_prefix,
        device=args.device,
    )
    elapsed = time.time() - start

    print(f"Channel: {args.channel}, Matcher: {args.matcher}")
    print(f"Keypoints A: {len(result['keypointsA'])}, Keypoints B: {len(result['keypointsB'])}")
    print(f"Matches: {result['num_matches']}")
    print(f"Elapsed: {elapsed:.2f}s")

    if args.output:
        save_result(result, args.output)
        print(f"Saved result to: {args.output}")

    if args.visualize:
        imgA = build_channel(patch_a, args.scene_a_prefix, args.channel)
        imgB = build_channel(patch_b, args.scene_b_prefix, args.channel)
        visualize(imgA, imgB, result, args.visualize)
        print(f"Saved visualization to: {args.visualize}")

    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-format", default="npz", choices=["npz", "geotiff"],
                   help="'npz' for the project's dataset format (default); 'geotiff' to "
                        "match raw, freshly downloaded per-band GeoTIFF/JP2 scenes that "
                        "were never packaged into .npz (requires 'rasterio' for proper "
                        "georeferenced reading; falls back to plain image reading if "
                        "rasterio isn't installed)")

    # --input-format npz
    p.add_argument("--patch-a", default=None, help="[npz] Path to the .npz file holding scene A")
    p.add_argument("--patch-b", default=None,
                   help="[npz] Path to the .npz file holding scene B (defaults to --patch-a, "
                        "the usual case where both scenes live in the same file)")
    p.add_argument("--scene-a-prefix", default="a")
    p.add_argument("--scene-b-prefix", default="b")

    # --input-format geotiff
    p.add_argument("--scene-a-dir", default=None,
                   help="[geotiff] Directory containing scene A's per-band GeoTIFF/JP2 files")
    p.add_argument("--scene-b-dir", default=None,
                   help="[geotiff] Directory containing scene B's per-band GeoTIFF/JP2 files")

    p.add_argument("--channel", default="B03", choices=SUPPORTED_CHANNELS)
    p.add_argument("--matcher", default="aliked-lightglue", choices=SUPPORTED_MATCHERS)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output", default=None, help="Path to save matches (.json or .npz)")
    p.add_argument("--visualize", default=None, help="Path to save a match visualization (.png)")
    args = p.parse_args()

    if args.input_format == "npz":
        if not args.patch_a:
            p.error("--patch-a is required when --input-format=npz")
        if args.patch_b is None:
            args.patch_b = args.patch_a
    else:
        if not args.scene_a_dir or not args.scene_b_dir:
            p.error("--scene-a-dir and --scene-b-dir are required when --input-format=geotiff")
        # scene_a_prefix/scene_b_prefix are fixed by combine_scenes() in this mode
        args.scene_a_prefix, args.scene_b_prefix = "a", "b"

    return args


if __name__ == "__main__":
    run(parse_args())
