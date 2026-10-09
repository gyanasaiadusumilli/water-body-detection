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

    pivot = df.pivot_table(index="model", columns="sensor", values="iou_water").reindex(
        [m for m in ORDER if m in set(df["model"])])
    ax = pivot.plot.barh(figsize=(8, 5))
    ax.set_xlabel("Water IoU (test set)")
    ax.set_title("Water body detection: model comparison")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS / "comparison_iou.png", dpi=150)
    print(f"Saved {RESULTS / 'comparison.csv'}, comparison.md and comparison_iou.png")


if __name__ == "__main__":
    main()
