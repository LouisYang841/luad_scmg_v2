#!/usr/bin/env python3
"""76_diagnose_depth_confound.py

Why is the GSE148071 positive control inverted (malignant epithelium scores LOWER
arm-level deviation than annotated normal alveolar epithelium)?

Two candidate explanations give opposite advice about what to do next:

  A) Depth/complexity confound. Malignant cells are deeper (fewer zeros), so their
     log1p(CPM) profile is compressed toward the mean and every |arm| shrinks. This
     is a fixable normalisation artefact -> match depth and the signal should appear.

  B) No real arm-level signal in this data (10x 3', ~1.1k genes/cell). Then no amount
     of depth matching helps, and the CNV route should be closed with a negative result.

This script separates them without re-running the CNV inference:
  1. per-cell detected-gene count in the CNV gene space, normal vs malignant;
  2. correlation of that count with cnv_load (the smoking gun for A);
  3. AUROC restricted to the normal cells' depth range (what A predicts should improve);
  4. same, after regressing log(depth) out of cnv_load (removes A's linear component);
  5. per-span bootstrap: does any single arm carry a group difference beyond noise?
"""
import os
import sys

import cupy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

DATASET = "GSE148071"
CHROMS = [str(i) for i in range(1, 23)] + ["X"]

# ---- the exact gene space 71_infer_cnv.py scores on ----
pos = pd.read_csv(f"{L.RES}/gene_positions_gencode44.csv")
pos = pos[pos.gene_type == "protein_coding"].copy()
std = np.load(f"{L.DAT}/std_genes.npy", allow_pickle=True).astype(str)
pm = pos.set_index("ensg")
keep = [g for g in std if g in pm.index]
cols_std = np.array([np.where(std == g)[0][0] for g in keep])
gp = pm.loc[keep].reset_index()
gp["chrom"] = gp.chrom.str[3:]
gp["col"] = cols_std
gp = gp[gp.chrom.isin(CHROMS)].copy()
gp["cnum"] = gp.chrom.map({c: i for i, c in enumerate(CHROMS)})
gp = gp.sort_values(["cnum", "start"]).reset_index(drop=True)
cols = gp.col.values.astype(np.int64)

order = pd.read_csv(f"{L.DAT}/cell_order.csv")
rows = order.index[order.dataset == DATASET].to_numpy()

X = L.load_X_std()
print(f"counting detected genes for {len(rows):,} cells over {len(cols):,} CNV genes ...", flush=True)
n_det = np.zeros(len(rows), np.float32)
for s in range(0, len(rows), 8000):
    b = rows[s:s + 8000]
    n_det[s:s + len(b)] = cp.asnumpy((X[np.ix_(b, cols)] > 0).sum(1)).astype(np.float32)
del X

meta = pd.read_csv(f"{L.RES}/cnv/{DATASET}/cnv_per_cell.csv", index_col=0)
d = pd.DataFrame({"n_det": n_det}, index=order.loc[rows, "barcode"].astype(str).values)
d = d.join(meta[["cnv_load", "compartment_clean", "patient_id"]])
d["pid"] = d.patient_id.astype(str).str.replace(f"{DATASET}_P", "", regex=False).astype(int)
d["grp"] = np.where(d.pid <= 5, "NORMAL(P1-5)", "MALIGNANT(P6-61)")
epi = d[d.compartment_clean.isin(["malignant_epithelium", "normal_alveolar_epithelial"])].copy()
print(f"epithelial cells: {len(epi):,}\n")

print("=== 1. detected-gene count by group ===")
print(epi.groupby("grp").n_det.describe().round(0).to_string())

print("\n=== 2. Pearson r(n_det, cnv_load) ===")
for g, s in epi.groupby("grp"):
    print(f"  {g:20s}: r = {np.corrcoef(s.n_det, s.cnv_load)[0, 1]:+.3f}  (n={len(s):,})")
print(f"  {'both groups pooled':20s}: r = {np.corrcoef(epi.n_det, epi.cnv_load)[0, 1]:+.3f}")

nrm = epi[epi.grp == "NORMAL(P1-5)"]
mal = epi[epi.grp == "MALIGNANT(P6-61)"]


def report(tag, m, n):
    y = np.r_[np.ones(len(m)), np.zeros(len(n))]
    s = np.r_[m.cnv_load.values, n.cnv_load.values]
    print(f"  {tag:38s} AUROC = {roc_auc_score(y, s):.4f}   "
          f"median load  mal={np.median(m.cnv_load):.4f} (n={len(m):,})  "
          f"norm={np.median(n.cnv_load):.4f} (n={len(n):,})")


print("\n=== 3. AUROC restricted to overlapping depth ===")
report("all cells (unmatched)", mal, nrm)
lo, hi = nrm.n_det.quantile([0.05, 0.95])
mm = mal[mal.n_det.between(lo, hi)]
print(f"  normal 5-95% n_det range: {lo:.0f}-{hi:.0f}; malignant inside: {len(mm):,}/{len(mal):,}")
report("malignant restricted to normal depth", mm, nrm)
# and the reverse direction: restrict normal to the malignant range
lo2, hi2 = mal.n_det.quantile([0.25, 0.75])
nn = nrm[nrm.n_det.between(lo2, hi2)]
print(f"  malignant IQR n_det range: {lo2:.0f}-{hi2:.0f}; normal inside: {len(nn):,}/{len(nrm):,}")
report("normal restricted to malignant depth", mal, nn)

print("\n=== 4. AUROC after removing the linear depth component ===")
z = np.log(epi.n_det.values)
A = np.c_[np.ones_like(z), z, z ** 2]
beta, *_ = np.linalg.lstsq(A, epi.cnv_load.values, rcond=None)
epi["cnv_load_adj"] = epi.cnv_load.values - A @ beta
print(f"  fitted: cnv_load ~ 1 + log(n_det) + log(n_det)^2")
report("depth-adjusted (pooled fit)",
       epi[(epi.grp == "MALIGNANT(P6-61)")], epi[(epi.grp == "NORMAL(P1-5)")])

print("\n=== 5. per-arm group difference on the depth-adjusted consensus ===")
arms = pd.read_csv(f"{L.RES}/cnv/{DATASET}/cnv_arm_matrix.csv.gz", index_col=0, nrows=5).columns.tolist()
am = pd.read_csv(f"{L.RES}/cnv/{DATASET}/cnv_arm_matrix.csv.gz", index_col=0)
am = am.loc[am.index.isin(epi.index)]
adj = am.copy()
z2 = np.log(epi.loc[am.index, "n_det"].values)
A2 = np.c_[np.ones_like(z2), z2, z2 ** 2]
for a in arms:
    b, *_ = np.linalg.lstsq(A2, am[a].values, rcond=None)
    adj[a] = am[a].values - A2 @ b
g = adj.groupby(epi.loc[am.index, "grp"].values).median()
diff = (g.loc["MALIGNANT(P6-61)"] - g.loc["NORMAL(P1-5)"]).sort_values()
print("  most negative (malignant lower):", ", ".join(f"{a}{v:+.3f}" for a, v in diff.head(6).items()))
print("  most positive (malignant higher):", ", ".join(f"{a}{v:+.3f}" for a, v in diff.tail(6).items()))
LUAD_AMP = ["1q", "5p", "7p", "7q", "8q", "20q"]
LUAD_DEL = ["3p", "4q", "8p", "9p", "9q", "10q", "13q", "18q"]
amp = diff[[a for a in LUAD_AMP if a in diff.index]]
dele = diff[[a for a in LUAD_DEL if a in diff.index]]
print(f"  mean over canonical LUAD amp arms: {amp.mean():+.4f}")
print(f"  mean over canonical LUAD del arms: {dele.mean():+.4f}")
print("\n  A real CNV signal needs amp arms > 0 AND del arms < 0 (opposite signs).")
