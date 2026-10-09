"""Cut exported GeoTIFF scenes into tiles and assign a spatial train/val/test split.

Neighbouring tiles look alike, so a purely random tile split leaks information into the
test set. Tiles are grouped into spatial blocks (block_tiles x block_tiles) and whole
blocks are assigned to a split.

Usage:
    python -m src.prepare_data --sensor sentinel2
    python -m src.prepare_data --sensor landsat8
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio

from src.features import IGNORE


def tile_scene(path, tile, block_tiles, max_ignore, rng, split_probs):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)                 # (7, H, W)
    if arr.shape[0] != 7:
        raise ValueError(f"{path}: expected 7 bands, found {arr.shape[0]}")
    X = np.nan_to_num(arr[:6], nan=0.0)
    y = np.nan_to_num(arr[6], nan=IGNORE).astype(np.uint8)
    _, H, W = X.shape

    block_split = {}
    out_X, out_y, out_split = [], [], []
    for i in range(0, H - tile + 1, tile):
        for j in range(0, W - tile + 1, tile):
            yt = y[i:i + tile, j:j + tile]
            if (yt == IGNORE).mean() > max_ignore:
                continue
            block = (i // (tile * block_tiles), j // (tile * block_tiles))
            if block not in block_split:
                block_split[block] = rng.choice(["train", "val", "test"], p=split_probs)
            out_X.append(X[:, i:i + tile, j:j + tile])
            out_y.append(yt)
            out_split.append(block_split[block])
    return out_X, out_y, out_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", required=True, choices=["sentinel2", "landsat8"])
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--block-tiles", type=int, default=4)
    p.add_argument("--max-ignore", type=float, default=0.5, help="Drop tiles with more no-data than this")
    p.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    scenes = sorted((Path(args.raw_dir) / args.sensor).glob("*.tif"))
    if not scenes:
        raise SystemExit(f"No GeoTIFFs found in {Path(args.raw_dir) / args.sensor}")
    rng = np.random.default_rng(args.seed)

    Xs, ys, splits = [], [], []
    for path in scenes:
        X, y, s = tile_scene(path, args.tile, args.block_tiles, args.max_ignore, rng, args.split)
        print(f"{path.name}: {len(X)} tiles")
        Xs += X
        ys += y
        splits += s

    X = np.stack(Xs).astype(np.float16)
    y = np.stack(ys)
    split = np.array(splits)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{args.sensor}.npz", X=X, y=y, split=split)

    for name in ["train", "val", "test"]:
        m = split == name
        labels = y[m]
        valid = labels != IGNORE
        water = (labels[valid] == 1).mean() if valid.any() else float("nan")
        print(f"{name:5s}: {m.sum():5d} tiles | water fraction {water:.3f}")
    print(f"Saved {out / (args.sensor + '.npz')}")


if __name__ == "__main__":
    main()
