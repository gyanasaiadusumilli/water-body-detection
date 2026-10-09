"""Train and evaluate the classical models (pixel-wise) on one sensor.

Each pixel is described by 10 features: 6 reflectance bands + NDWI, MNDWI, NDVI, AWEI_sh.
Also scores two index-threshold baselines that need no training.

Usage:
    python -m src.train_ml --sensor sentinel2
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common import compute_metrics, load_tiles, save_result, test_pixel_sample
from src.features import FEATURES, IGNORE, add_indices, tiles_to_pixels


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


# (model, max training pixels). SVM and KNN get smaller subsets because they scale poorly.
def get_models(seed):
    return {
        "logistic_regression": (make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, class_weight="balanced")), 300_000),
        "random_forest": (RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1, class_weight="balanced", random_state=seed), 300_000),
        "gradient_boosting": (HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed), 300_000),
        "knn": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, n_jobs=-1)), 50_000),
        "svm_rbf": (make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=10, gamma="scale", class_weight="balanced")), 20_000),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", required=True, choices=["sentinel2", "landsat8"])
    p.add_argument("--models", nargs="*", default=None, help="Subset of models to run")
    p.add_argument("--max-test-pixels", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    X, y, split = load_tiles(args.sensor)
    rng = np.random.default_rng(args.seed)
    Xtr, ytr = tiles_to_pixels(X[split == "train"], y[split == "train"])
    print(f"Training pixels available: {len(ytr):,} (water fraction {ytr.mean():.3f})")

    # Fixed test sample shared with the CNN evaluation
    y_test_tiles = y[split == "test"]
    idx = test_pixel_sample(y_test_tiles, args.max_test_pixels)
    test_feats = add_indices(X[split == "test"], axis=1).transpose(0, 2, 3, 1).reshape(-1, len(FEATURES))
    Xte, yte = test_feats[idx], y_test_tiles.reshape(-1)[idx].astype(np.int64)
    assert (yte != IGNORE).all()

    # Index-threshold baselines
    ndwi, mndwi = FEATURES.index("ndwi"), FEATURES.index("mndwi")
    save_result(args.sensor, "ndwi_threshold_0", compute_metrics(yte, (Xte[:, ndwi] > 0).astype(int)))
    t = otsu_threshold(Xtr[:, mndwi])
    save_result(args.sensor, "mndwi_otsu", compute_metrics(yte, (Xte[:, mndwi] > t).astype(int)),
                {"threshold": t})

    model_dir = Path("models") / args.sensor
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, (model, max_train) in get_models(args.seed).items():
        if args.models and name not in args.models:
            continue
        n = min(max_train, len(ytr))
        sel = rng.choice(len(ytr), n, replace=False)
        t0 = time.time()
        model.fit(Xtr[sel], ytr[sel])
        train_s = time.time() - t0
        t0 = time.time()
        pred = model.predict(Xte)
        pred_s = time.time() - t0
        save_result(args.sensor, name, compute_metrics(yte, pred),
                    {"train_pixels": int(n), "train_seconds": round(train_s, 1),
                     "predict_seconds": round(pred_s, 1)})
        joblib.dump(model, model_dir / f"{name}.joblib")

        if name == "random_forest":
            imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda kv: -kv[1])
            print("  Random forest feature importance: " + ", ".join(f"{f}={v:.3f}" for f, v in imp))


if __name__ == "__main__":
    main()
