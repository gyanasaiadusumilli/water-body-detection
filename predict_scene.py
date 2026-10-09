"""Produce a water mask for a whole scene and save it as a GeoTIFF plus a PNG preview.

Works with the CNN checkpoint or any saved classical model.

Usage:
    python -m src.predict_scene data/raw/sentinel2/scene.tif --model models/sentinel2/cnn_unet.pt
    python -m src.predict_scene data/raw/landsat8/scene.tif --model models/landsat8/random_forest.joblib
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402

from src.cnn import UNet  # noqa: E402
from src.features import add_indices  # noqa: E402


@torch.no_grad()
def predict_cnn(feats, ckpt_path, window=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet(ckpt["in_channels"], ckpt["base"], ckpt["depth"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    x = (feats - ckpt["mean"][:, None, None]) / ckpt["std"][:, None, None]

    _, H, W = x.shape
    ph, pw = (-H) % window, (-W) % window
    x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
    prob = np.zeros(x.shape[1:], dtype=np.float32)
    for i in range(0, x.shape[1], window):
        for j in range(0, x.shape[2], window):
            patch = torch.from_numpy(x[None, :, i:i + window, j:j + window].astype(np.float32)).to(device)
            prob[i:i + window, j:j + window] = torch.sigmoid(model(patch))[0, 0].cpu().numpy()
    return prob[:H, :W]


def predict_ml(feats, model_path, chunk=500_000):
    model = joblib.load(model_path)
    C, H, W = feats.shape
    flat = feats.reshape(C, -1).T
    pred = np.concatenate([model.predict(flat[i:i + chunk]) for i in range(0, len(flat), chunk)])
    return pred.reshape(H, W).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene", help="GeoTIFF with at least the 6 reflectance bands first")
    p.add_argument("--model", required=True, help=".pt (CNN) or .joblib (classical model)")
    p.add_argument("--out-dir", default="predictions")
    args = p.parse_args()

    with rasterio.open(args.scene) as src:
        bands = np.nan_to_num(src.read(list(range(1, 7))).astype(np.float32))
        profile = src.profile
    feats = add_indices(bands, axis=0)

    if args.model.endswith(".pt"):
        mask = (predict_cnn(feats, args.model) >= 0.5).astype(np.uint8)
    else:
        mask = (predict_ml(feats, args.model) >= 0.5).astype(np.uint8)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.scene).stem}_{Path(args.model).stem}"
    profile.update(count=1, dtype="uint8", nodata=None, compress="lzw")
    with rasterio.open(out / f"{stem}_water.tif", "w", **profile) as dst:
        dst.write(mask, 1)

    rgb = np.clip(bands[[2, 1, 0]].transpose(1, 2, 0) * 3.5, 0, 1)   # true color, brightened
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("True color")
    axes[1].imshow(rgb)
    axes[1].imshow(np.ma.masked_where(mask == 0, mask), cmap="cool", alpha=0.6)
    axes[1].set_title(f"Detected water ({mask.mean():.1%} of area)")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out / f"{stem}_preview.png", dpi=150)
    print(f"Saved {out / (stem + '_water.tif')} and preview PNG")


if __name__ == "__main__":
    main()
