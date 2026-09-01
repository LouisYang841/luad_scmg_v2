#!/usr/bin/env python3
"""Validate CNV scores against annotated malignant and held-out normal epithelium."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import luadlib as L

parser = argparse.ArgumentParser(description="Plot a CNV positive-control comparison")
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

out = f"{L.RES}/cnv/{args.dataset}"
per_cell = f"{out}/cnv_per_cell.csv"
summary_path = f"{out}/cnv_summary.json"

cells = pd.read_csv(per_cell)
summary = json.load(open(summary_path))
malignant = cells.loc[
    cells.compartment_clean.eq("malignant_epithelium"), "cnv_load"
].to_numpy()
heldout = cells.loc[
    cells.compartment_clean.eq("normal_alveolar_epithelial")
    & cells.is_null_holdout.astype(bool),
    "cnv_load",
].to_numpy()
if len(malignant) < 20 or len(heldout) < 20:
    raise RuntimeError("positive-control groups must each contain at least 20 cells")

labels = np.r_[np.ones(len(malignant)), np.zeros(len(heldout))]
scores = np.r_[malignant, heldout]
auroc = float(roc_auc_score(labels, scores))
threshold = float(summary["thr_q99_matched_null"])

upper = float(np.quantile(scores, 0.995))
bins = np.linspace(0, upper, 70)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].hist(heldout, bins=bins, density=True, histtype="step", linewidth=2,
             label=f"Held-out normal (n={len(heldout):,})")
axes[0].hist(malignant, bins=bins, density=True, histtype="step", linewidth=2,
             label=f"Malignant (n={len(malignant):,})")
axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1,
                label="Held-out normal q99")
axes[0].set(xlabel="CNV load (mean |arm|)", ylabel="Density",
            title=f"{args.dataset}: AUROC = {auroc:.3f}")
axes[0].legend(frameon=False, fontsize=8)

parts = axes[1].violinplot([heldout, malignant], showmeans=False, showmedians=True)
for body in parts["bodies"]:
    body.set_alpha(0.65)
axes[1].set_xticks([1, 2], ["Held-out normal", "Malignant"])
axes[1].set_ylabel("CNV load (mean |arm|)")
axes[1].set_title("Positive-control score distributions")
axes[1].text(
    0.03, 0.97,
    f"medians: {np.median(heldout):.3f} vs {np.median(malignant):.3f}\n"
    f"called: {(heldout > threshold).mean():.3%} vs {(malignant > threshold).mean():.3%}",
    transform=axes[1].transAxes, va="top", fontsize=9,
)
fig.tight_layout()
figure_path = f"{out}/positive_control_cnv_load.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)

metrics = {
    "dataset": args.dataset,
    "malignant_n": int(len(malignant)),
    "heldout_normal_n": int(len(heldout)),
    "auroc": auroc,
    "malignant_median": float(np.median(malignant)),
    "heldout_normal_median": float(np.median(heldout)),
    "threshold_q99_heldout_normal": threshold,
    "malignant_fraction_called": float((malignant > threshold).mean()),
    "heldout_normal_fraction_called": float((heldout > threshold).mean()),
}
metrics_path = f"{out}/positive_control_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
print(f"wrote {os.path.relpath(figure_path, L.ROOT)}")
