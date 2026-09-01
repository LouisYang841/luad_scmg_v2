#!/usr/bin/env python3
"""77_check_arm_biology.py

The GSE148071 positive control now clears AUROC > 0.7 (0.757). But an AUROC on a
patient-level comparison can be produced by any systematic difference between two
groups of patients -- batch, chemistry, dissociation, depth -- not just copy number.

Copy number has a signature nothing else has: it is arm-specific, it recurs across
independent patients, and the recurrent arms are the ones LUAD is known to hit.
This script tests for that signature on the per-patient consensus profiles.

Test: if the malignant group carries real CNAs, the canonical LUAD amplification
arms should shift UP and the canonical deletion arms should shift DOWN -- i.e. the
two sets must separate with OPPOSITE signs. A permutation test over arm labels
gives the null for "the arms that move are just the arms that move".

Also re-checks the depth confound against the current (post-fix) scores: if the
group difference survives matching on detected-gene count, depth is not the driver.
"""
import os
import sys

import cupy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

DATASET = "GSE148071"
CHROMS = [str(i) for i in range(1, 23)] + ["X"]
OUT = f"{L.RES}/cnv/{DATASET}"

# Canonical LUAD arm-level events (TCGA LUAD / Broad, recurrent arm-level CNAs).
LUAD_AMP = ["1q", "5p", "7p", "7q", "8q", "20q"]
LUAD_DEL = ["3p", "4q", "8p", "9p", "9q", "10q", "13q", "18q"]

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- consensus ----
c = pd.read_csv(f"{OUT}/cnv_arm_by_patient_epithelium.csv")
arms = [a for a in c.columns if a not in ("dataset", "patient_id", "n_cells")]
c["pid"] = c.patient_id.astype(str).str.replace(f"{DATASET}_P", "", regex=False).astype(int)
c["grp"] = np.where(c.pid <= 5, "NORMAL", "MALIGNANT")
# tiny patients (a handful of cells) make the median meaningless
c = c[c.n_cells >= 30]
print(f"patients with >=30 epithelial cells: {len(c)} "
      f"(NORMAL={int((c.grp == 'NORMAL').sum())}, MALIGNANT={int((c.grp == 'MALIGNANT').sum())})")

mal = c[c.grp == "MALIGNANT"][arms].to_numpy()
nor = c[c.grp == "NORMAL"][arms].to_numpy()


def robust_z(a, b):
    """median difference, scaled by a robust pooled spread across patients"""
    diff = np.median(a, 0) - np.median(b, 0)
    spread = 1.4826 * np.median(np.abs(a - np.median(a, 0)), 0) / max(np.sqrt(a.shape[0]), 2) \
        + 1.4826 * np.median(np.abs(b - np.median(b, 0)), 0) / max(np.sqrt(b.shape[0]), 2)
    return diff, diff / np.maximum(spread, 1e-9)


diff, z = robust_z(mal, nor)
idx = {a: i for i, a in enumerate(arms)}
amp_i = [idx[a] for a in LUAD_AMP if a in idx]
del_i = [idx[a] for a in LUAD_DEL if a in idx]

print("\n=== 1. canonical LUAD arms: median(malignant) - median(normal), per patient consensus ===")
for a in LUAD_AMP:
    if a in idx:
        print(f"  AMP {a:>4}: {diff[idx[a]]:+.4f}   (z = {z[idx[a]]:+.2f})")
for a in LUAD_DEL:
    if a in idx:
        print(f"  DEL {a:>4}: {diff[idx[a]]:+.4f}   (z = {z[idx[a]]:+.2f})")

stat = z[amp_i].mean() - z[del_i].mean()
print(f"\n  separation statistic  mean z(AMP arms) - mean z(DEL arms) = {stat:+.3f}")
print("  (a real CNA signature needs this to be clearly POSITIVE)")

# -------------------------------------------------- permutation test on labels ----
allz = z.copy()
n_perm = 20000
perm = np.empty(n_perm)
for i in range(n_perm):
    pick = rng.permutation(len(allz))
    perm[i] = allz[pick[:len(amp_i)]].mean() - allz[pick[len(amp_i):len(amp_i) + len(del_i)]].mean()
p_two = (np.sum(np.abs(perm) >= abs(stat)) + 1) / (n_perm + 1)
p_one = (np.sum(perm >= stat) + 1) / (n_perm + 1)
print(f"  permutation null over {n_perm:,} arm relabellings: "
      f"mean={perm.mean():+.3f} sd={perm.std():.3f}")
print(f"  p(one-sided, AMP>DEL) = {p_one:.4f}    p(two-sided) = {p_two:.4f}")

print("\n=== 2. arms that actually move (|z| ranked) ===")
order_ = np.argsort(z)
print("  most DOWN in malignant:", ", ".join(f"{arms[i]}{z[i]:+.2f}" for i in order_[:8]))
print("  most UP   in malignant:", ", ".join(f"{arms[i]}{z[i]:+.2f}" for i in order_[-8:][::-1]))

print("\n=== 3. recurrence: how many malignant patients clear a normal-based threshold ===")
# per arm, threshold = (max of the 5 normal patients) with a small margin
hi = np.max(nor, 0) + 0.02
lo = np.min(nor, 0) - 0.02
up_frac = (mal > hi).mean(0)
dn_frac = (mal < lo).mean(0)
rec = pd.DataFrame({"arm": arms,
                    "frac_mal_above_all_normal": up_frac.round(3),
                    "frac_mal_below_all_normal": dn_frac.round(3)})
rec["canonical"] = ["AMP" if a in LUAD_AMP else ("DEL" if a in LUAD_DEL else "") for a in arms]
print(rec.sort_values("frac_mal_above_all_normal", ascending=False).head(10).to_string(index=False))
print("  ...")
print(rec.sort_values("frac_mal_below_all_normal", ascending=False).head(10).to_string(index=False))
print(f"\n  mean recurrence, canonical AMP arms (above): {up_frac[amp_i].mean():.3f}")
print(f"  mean recurrence, canonical DEL arms (below): {dn_frac[del_i].mean():.3f}")
print(f"  mean recurrence, all other arms  (above/below): "
      f"{np.mean(np.delete(up_frac, amp_i + del_i)):.3f} / "
      f"{np.mean(np.delete(dn_frac, amp_i + del_i)):.3f}")

# ------------------------------------------------------- depth confound recheck ----
print("\n=== 4. depth confound against the current (post-fix) scores ===")
from sklearn.metrics import roc_auc_score

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
n_det = np.zeros(len(rows), np.float32)
for s in range(0, len(rows), 8000):
    b = rows[s:s + 8000]
    n_det[s:s + len(b)] = cp.asnumpy((X[np.ix_(b, cols)] > 0).sum(1)).astype(np.float32)
del X

meta = pd.read_csv(f"{OUT}/cnv_per_cell.csv", index_col=0)
d = pd.DataFrame({"n_det": n_det}, index=order.loc[rows, "barcode"].astype(str).values)
d = d.join(meta[["cnv_load", "compartment_clean"]])
epi = d[d.compartment_clean.isin(["malignant_epithelium", "normal_alveolar_epithelial"])]
m = epi[epi.compartment_clean == "malignant_epithelium"]
n = epi[epi.compartment_clean == "normal_alveolar_epithelial"]
print(f"  r(n_det, cnv_load) pooled = {np.corrcoef(epi.n_det, epi.cnv_load)[0, 1]:+.3f}")


def auc(mm, nn):
    y = np.r_[np.ones(len(mm)), np.zeros(len(nn))]
    s = np.r_[mm.cnv_load.values, nn.cnv_load.values]
    return roc_auc_score(y, s)


print(f"  AUROC all cells                       = {auc(m, n):.4f}")
lo_d, hi_d = n.n_det.quantile([0.05, 0.95])
mm = m[m.n_det.between(lo_d, hi_d)]
print(f"  AUROC, malignant matched to normal depth ({lo_d:.0f}-{hi_d:.0f}, n={len(mm):,}) = {auc(mm, n):.4f}")
# stratify: AUROC within depth deciles
qs = np.quantile(epi.n_det, np.linspace(0, 1, 6))
within = []
for i in range(5):
    sel_m = m[(m.n_det >= qs[i]) & (m.n_det <= qs[i + 1])]
    sel_n = n[(n.n_det >= qs[i]) & (n.n_det <= qs[i + 1])]
    if len(sel_m) >= 50 and len(sel_n) >= 50:
        within.append((qs[i], auc(sel_m, sel_n), len(sel_m), len(sel_n)))
print("  AUROC within depth quintiles:")
for a, u, nm, nn_ in within:
    print(f"    n_det>={a:6.0f}: AUROC={u:.4f}  (mal={nm:,}, norm={nn_:,})")
