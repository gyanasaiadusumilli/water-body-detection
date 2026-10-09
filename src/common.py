"""Shared helpers: loading tiles, the fixed test-pixel sample, and metrics."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, jaccard_score,
                             precision_score, recall_score)

from src.features import IGNORE


def load_tiles(sensor, data_dir="data/processed"):
    d = np.load(Path(data_dir) / f"{sensor}.npz")
    return d["X"].astype(np.float32), d["y"], d["split"]


def test_pixel_sample(y_test: np.ndarray, max_pixels: int = 500_000, seed: int = 0) -> np.ndarray:
    """Fixed random subset of valid test pixels (flat indices into y_test.reshape(-1)).

    Every model, classical or CNN, is scored on exactly these pixels so the comparison is fair.
    Slow models such as SVM and KNN cannot predict millions of pixels in reasonable time.
    """
    flat = y_test.reshape(-1)
    valid = np.flatnonzero(flat != IGNORE)
    if len(valid) <= max_pixels:
        return valid
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(valid, max_pixels, replace=False))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "iou_water": float(jaccard_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_pixels": int(len(y_true)),
    }


def save_result(sensor, model_name, metrics, extra=None, results_dir="results"):
    out = Path(results_dir) / sensor
    out.mkdir(parents=True, exist_ok=True)
    payload = {"sensor": sensor, "model": model_name, **metrics, **(extra or {})}
    (out / f"{model_name}.json").write_text(json.dumps(payload, indent=2))
    print(f"[{sensor}] {model_name:22s} acc {metrics['accuracy']:.4f}  F1 {metrics['f1']:.4f}  "
          f"IoU {metrics['iou_water']:.4f}  kappa {metrics['kappa']:.4f}")
