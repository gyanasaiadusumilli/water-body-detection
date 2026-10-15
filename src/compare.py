"""Collect all results into a comparison table and chart.

Usage:
    python -m src.compare
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")
ORDER = ["ndwi_threshold_0", "mndwi_otsu", "logistic_regression", "knn", "svm_rbf",
         "random_forest", "gradient_boosting", "cnn_unet"]


def main():
    rows = [json.loads(f.read_text()) for f in RESULTS.glob("*/*.json") if f.name != "cnn_history.json"]
    if not rows:
        raise SystemExit("No results found. Run train_ml and train_cnn first.")
    df = pd.DataFrame(rows)
    df["order"] = df["model"].map({m: i for i, m in enumerate(ORDER)}).fillna(len(ORDER))
    df = df.sort_values(["sensor", "order"])
    cols = ["sensor", "model", "accuracy", "precision", "recall", "f1", "iou_water", "kappa"]

    table = df[cols].round(4)
    table.to_csv(RESULTS / "comparison.csv", index=False)
    (RESULTS / "comparison.md").write_text(table.to_markdown(index=False))
    print(table.to_string(index=False))

    names = {"ndwi_threshold_0": "NDWI > 0", "mndwi_otsu": "MNDWI (Otsu)", "logistic_regression": "Logistic regression",
             "knn": "KNN", "svm_rbf": "SVM (RBF)", "random_forest": "Random forest",
             "gradient_boosting": "Gradient boosting", "cnn_unet": "U-Net CNN"}
    sensors = [s for s in ["sentinel2", "landsat8"] if s in set(df["sensor"])]
    titles = {"sentinel2": "Sentinel-2 (10 m)", "landsat8": "Landsat-8 (30 m)"}
    lo = max(0.0, df["iou_water"].min() - 0.01)
    fig, axes = plt.subplots(1, len(sensors), figsize=(5.2 * len(sensors), 4.6), sharey=True, squeeze=False)
    for ax, sensor in zip(axes[0], sensors):
        d = df[df["sensor"] == sensor].sort_values("order")
        best = d["iou_water"].max()
        colors = ["#0B7C8A" if v == best else ("#B9C6C9" if m in ("ndwi_threshold_0", "mndwi_otsu") else "#6F8A92")
                  for m, v in zip(d["model"], d["iou_water"])]
        y = range(len(d))
        ax.barh(list(y), d["iou_water"] - lo, left=lo, color=colors, height=0.68)
        for yi, v in zip(y, d["iou_water"]):
            ax.text(v + 0.0006, yi, f"{v:.3f}", va="center", fontsize=9, fontweight="bold" if v == best else "normal")
        ax.set_yticks(list(y))
        ax.set_yticklabels([names.get(m, m) for m in d["model"]])
        ax.invert_yaxis()
        ax.set_xlim(lo, 1.0)
        ax.set_title(titles.get(sensor, sensor), fontsize=11)
        ax.set_xlabel("Water IoU on the test set (higher is better)")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Water body detection: model comparison", fontsize=13)
    fig.text(0.5, 0.005, "Light grey: index baselines. Teal: best model per sensor. Note the zoomed x-axis.",
             ha="center", fontsize=8.5, color="#555")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(RESULTS / "comparison_iou.png", dpi=160)
    print(f"Saved {RESULTS / 'comparison.csv'}, comparison.md and comparison_iou.png")


if __name__ == "__main__":
    main()
