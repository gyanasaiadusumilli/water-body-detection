"""Train the U-Net CNN on one sensor, then score it on the same test pixels as the ML models.

Usage:
    python -m src.train_cnn --sensor sentinel2 --epochs 40
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.cnn import UNet
from src.common import compute_metrics, load_tiles, save_result, test_pixel_sample
from src.features import IGNORE, add_indices


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def channel_stats(X):
    """Per-channel mean/std over the training tiles (10 channels)."""
    mean = X.mean(axis=(0, 2, 3))
    std = X.std(axis=(0, 2, 3)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


class TileDataset(Dataset):
    def __init__(self, X, y, mean, std, augment=False):
        self.X = ((X - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]
        if self.augment:
            k = random.randint(0, 3)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if random.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy().astype(np.int64))


def masked_loss(logits, target):
    """BCE + soft Dice, both ignoring no-data pixels (label 255)."""
    logits = logits[:, 0]
    valid = target != IGNORE
    t = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits[valid], t[valid])
    p = torch.sigmoid(logits) * valid
    tv = t * valid
    dice = 1 - (2 * (p * tv).sum() + 1) / (p.sum() + tv.sum() + 1)
    return bce + dice


@torch.no_grad()
def predict_tiles(model, ds, device, batch_size=32):
    model.eval()
    out = []
    for x, _ in DataLoader(ds, batch_size, shuffle=False):
        out.append(torch.sigmoid(model(x.to(device)))[:, 0].cpu().numpy())
    return np.concatenate(out)


def iou(prob, y, thr=0.5):
    valid = y != IGNORE
    pred, true = prob[valid] >= thr, y[valid] == 1
    union = (pred | true).sum()
    return float((pred & true).sum() / union) if union else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", required=True, choices=["sentinel2", "landsat8"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--patience", type=int, default=8, help="Early stopping on validation IoU")
    p.add_argument("--max-test-pixels", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    X, y, split = load_tiles(args.sensor)
    X = add_indices(X, axis=1)                                      # (N, 10, H, W)
    tr, va, te = split == "train", split == "val", split == "test"
    mean, std = channel_stats(X[tr])
    train_ds = TileDataset(X[tr], y[tr], mean, std, augment=True)
    val_ds = TileDataset(X[va], y[va], mean, std)
    test_ds = TileDataset(X[te], y[te], mean, std)
    print(f"Device {device} | tiles train={tr.sum()} val={va.sum()} test={te.sum()}")

    model = UNet(in_channels=X.shape[1], base=args.base, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, int(np.ceil(len(train_ds) / args.batch_size))))
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True)

    ckpt_dir = Path("models") / args.sensor
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "cnn_unet.pt"
    best, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = masked_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
        val_iou = iou(predict_tiles(model, val_ds, device), y[va])
        history.append({"epoch": epoch, "train_loss": total / len(train_ds), "val_iou": val_iou})
        print(f"Epoch {epoch:02d} | loss {total / len(train_ds):.4f} | val IoU {val_iou:.4f} | {time.time() - t0:.0f}s")
        if val_iou > best:
            best, bad_epochs = val_iou, 0
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "base": args.base, "depth": args.depth, "in_channels": X.shape[1]}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    Path("results", args.sensor).mkdir(parents=True, exist_ok=True)
    Path("results", args.sensor, "cnn_history.json").write_text(json.dumps(history, indent=2))

    # Final test evaluation on the shared pixel sample
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    prob = predict_tiles(model, test_ds, device)
    idx = test_pixel_sample(y[te], args.max_test_pixels)
    y_true = y[te].reshape(-1)[idx].astype(np.int64)
    y_pred = (prob.reshape(-1)[idx] >= 0.5).astype(np.int64)
    save_result(args.sensor, "cnn_unet", compute_metrics(y_true, y_pred), {"best_val_iou": best})


if __name__ == "__main__":
    main()
