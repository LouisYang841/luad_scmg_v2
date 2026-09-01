#!/usr/bin/env python3
"""71_infer_cnv.py -- expression-based CNV inference, implemented on GPU ourselves.

Why not run inferCNV: it needs R + dense matrices (~30 GB here), expects gene SYMBOLS
(our standard space is ENSG), and its usual workflow would have us define the normal
reference from the very annotation whose correctness is under test.

Statistic (inferCNV's core):
  1. GENCODE protein_coding genes on chr1-22,X, ordered by position;
  2. row-centre each cell (per-cell median over used genes) to kill library/state offset;
  3. column-centre each gene by the median of REFERENCE cells;
  4. rolling mean over W consecutive genes, never crossing a chromosome;
  5. summarise to gene-bearing chromosome arms; score = mean |arm|.

Honest null: the reference is HALF of the adjacent-normal normal-epithelium cells
(per dataset, split by index parity); the OTHER half is scored but never contributed to
the reference. Thresholds come from that held-out half, so the null is measured on cells
of the exact same biological state rather than on the reference cells themselves
(which would make the null artificially tight and the calls circular).

Positive controls that must pass before anything else is believed:
GSE189357 and GSE148071 contain ANNOTATED malignant epithelium next to annotated normal
epithelium in the same dataset.
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
import cupy as cp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

DATASETS = ["GSE131907", "GSE189357", "GSE148071"]
parser = argparse.ArgumentParser(description="GPU expression-based CNV inference")
parser.add_argument("--datasets", nargs="+", default=DATASETS,
                    help="Datasets to score (space- or comma-separated)")
args = parser.parse_args()
requested = [ds for value in args.datasets for ds in value.split(",")]
unknown = sorted(set(requested) - set(DATASETS))
if unknown:
    parser.error(f"unknown datasets: {', '.join(unknown)}")
TARGETS = [ds for ds in DATASETS if ds in requested]
if not TARGETS:
    parser.error("at least one dataset is required")

OUT_ROOT = f"{L.RES}/cnv"
OUT = OUT_ROOT if TARGETS == DATASETS else f"{OUT_ROOT}/{'_'.join(TARGETS)}"
os.makedirs(OUT, exist_ok=True)
W = int(os.environ.get("CNV_W", "100"))
EXPL = 0.10
CHROMS = [str(i) for i in range(1, 23)] + ["X"]
print(f"datasets: {', '.join(TARGETS)}\noutput: {OUT}", flush=True)

order = pd.read_csv(f"{L.DAT}/cell_order.csv")
ann = pd.read_csv(f"{L.RES}/cell_annotations_clean.csv", index_col=0)
pos = pd.read_csv("/home/ubuntu/luad_scmg_v2/results/gene_positions_gencode44.csv")
pos = pos[pos.gene_type == "protein_coding"].copy()
std = np.load(f"{L.DAT}/std_genes.npy", allow_pickle=True).astype(str)

# ---- gene space, positional order ----
pm = pos.set_index("ensg")
keep = [g for g in std if g in pm.index]
cols = np.array([np.where(std == g)[0][0] for g in keep])
gp = pm.loc[keep].reset_index(drop=True)
gp["chrom"] = gp.chrom.str[3:]
gp["col"] = cols
gp = gp[gp.chrom.isin(CHROMS)].copy()
gp["cnum"] = gp.chrom.map({c: i for i, c in enumerate(CHROMS)})
gp = gp.sort_values(["cnum", "start"]).reset_index(drop=True)
n_genes = len(gp)
print(f"protein_coding genes on chr1-22,X: {n_genes}")

# ---- cell meta with clean compartments ----
meta = order.copy()
a = ann.reindex(meta.barcode.astype(str))
meta["compartment_clean"] = a["compartment_clean"].values
meta["site"] = a["tissue_site"].fillna("").values

def site_from_pid(p):
    p = str(p)
    for tag, lab in [("LUNG_N", "adjacent_normal"), ("LUNG_T", "tumor_sample"),
                     ("LN_", "lymph_node"), ("EBUS_", "lymph_node"),
                     ("EFFUSION", "pleural_effusion"), ("BRONCHOALVEOLAR", "airway")]:
        if "_" + tag in p or p.endswith(tag):
            return lab
    return ""
# 用样本名解析出的部位覆盖 tissue_site（GSE131907 的癌旁 vs 原发是配对设计的关键）
derived = meta.patient_id.map(site_from_pid)
meta["site"] = np.where(derived != "", derived, meta.site)
meta["site2"] = meta.site + "|" + a["tissue_site"].fillna("").values
print("\nsite x dataset:", pd.crosstab(meta.dataset, meta.site).to_string())

# ---- reference / held-out-null split ----
REF, NULLK = {}, {}
for ds in TARGETS:
    sel = meta[(meta.dataset == ds) & (meta.compartment_clean == "normal_alveolar_epithelial")]
    if ds == "GSE131907":
        sel = sel[sel.site.astype(str).str.contains("adjacent_normal")]
    elif ds == "GSE189357":
        # GSE189357 的上皮绝大多数来自原发切片；用其中标注为正常 AT2 的细胞作参考，
        # 并只用非淋巴结/非胸水部位，避免把转移灶当正常
        sel = sel[~sel.site.astype(str).str.contains("lymph_node|pleural_effusion")]
    idx = sel.index.values
    if len(idx) < 200:
        print(f"  !! {ds}: only {len(idx)} cells after site filter, skipped")
        continue
    REF[ds] = idx[idx % 2 == 0]        # even positional rows -> reference
    NULLK[ds] = idx[idx % 2 == 1]      # odd rows -> held-out matched-state null
    print(f"  {ds}: ref={len(REF[ds]):,}  null_holdout={len(NULLK[ds]):,}  (annotated normal epi)")
missing_ref = [ds for ds in TARGETS if ds not in REF]
if missing_ref:
    raise RuntimeError(f"no usable normal reference for: {', '.join(missing_ref)}")

X = L.load_X_std()          # 11.5 GB fp16, GPU-resident (do not use cp.load: pickled header)

def row_med(sel):
    """per-cell median over used genes (needs chunking to stay in memory)"""
    out = np.zeros(len(sel), np.float32)
    for s in range(0, len(sel), 20000):
        b = sel[s:s + 20000]
        out[s:s + len(b)] = cp.asnumpy(cp.median(X[np.ix_(b, cols)].astype(cp.float32), 1))
    return out

# ---- reference gene medians + expression filter ----
STAT = {}
for ds, idx in REF.items():
    B = X[np.ix_(idx, cols)].astype(cp.float32)
    expressed = cp.asnumpy((B > 0).mean(0)) >= EXPL
    B -= cp.median(B, 1, keepdims=True)
    med = cp.asnumpy(cp.median(B, 0)).astype(np.float32)
    STAT[ds] = (expressed, med)
    del B
print("reference gene medians done")

CH = gp.chrom.values
ARMV = gp.arm.values
ARMS = sorted(set(ARMV))
print(f"{len(ARMS)} chromosome arms")

def arm_scores(rows, ds):
    expressed, med = STAT[ds]
    gcols_local = np.where(expressed)[0]
    use_cols = cols[gcols_local]
    local_arms = ARMV[gcols_local]
    arm_of = np.array([ARMS.index(a) for a in local_arms])
    nA = len(ARMS)
    res = np.zeros((len(rows), nA), np.float32)
    cnt = np.bincount(arm_of, minlength=nA).astype(np.float32)
    for s in range(0, len(rows), 8000):
        b = rows[s:s + 8000]
        V = X[np.ix_(b, use_cols)].astype(cp.float32)
        V -= cp.median(V, 1, keepdims=True)
        V -= cp.array(med[gcols_local], dtype=cp.float32)[None, :]
        Sm = cp.zeros_like(V)
        for c in np.unique(CH[gcols_local]):
            m = CH[gcols_local] == c
            v = V[:, m]; n = v.shape[1]
            w = max(3, min(W, n))
            if n <= w:
                Sm[:, m] = v.mean(1, keepdims=True); continue
            pad_left, pad_right = (w - 1) // 2, w // 2
            vp = cp.pad(v, ((0, 0), (pad_left, pad_right)), mode="edge")
            cs = cp.cumsum(cp.pad(vp, ((0, 0), (1, 0))), 1)
            Sm[:, m] = (cs[:, w:] - cs[:, :-w]) / w
        agg = cp.zeros((len(b), nA), cp.float32)
        for ai in range(nA):
            agg[:, ai] = Sm[:, arm_of == ai].sum(1)
        res[s:s + len(b)] = cp.asnumpy(agg / cp.maximum(cp.asarray(cnt), 1)[None, :])
        del V, Sm, agg
    return res

ALL = meta.index.values
selected_rows = meta.index[meta.dataset.isin(TARGETS)].to_numpy()
AM = np.zeros((len(ALL), len(ARMS)), np.float32)
for ds in STAT:
    ii = meta.index[meta.dataset.values == ds]
    if len(ii) == 0: continue
    AM[ii] = arm_scores(ii, ds)
    print(f"  scored {ds}: {len(ii):,} cells", flush=True)
AMdf = pd.DataFrame(AM[selected_rows], columns=ARMS,
                    index=meta.loc[selected_rows, "barcode"].astype(str))
arm_path = f"{OUT}/cnv_arm_matrix.csv.gz"
arm_tmp = f"{arm_path}.tmp"
AMdf.to_csv(arm_tmp, compression="gzip")
os.replace(arm_tmp, arm_path)

# ---- honest null & thresholds ----
null_rows = np.concatenate([NULLK[ds] for ds in NULLK])
null_load = np.abs(AM[null_rows]).mean(1)
thr = float(np.quantile(null_load, 0.99))
print(f"matched-state held-out null: n={len(null_rows):,} median|arm|={np.median(null_load):.4f} "
      f"q95={np.quantile(null_load,0.95):.4f} q99={thr:.4f}")

meta["cnv_load"] = np.abs(AM).mean(1)
meta["cnv_amp"] = AM.clip(min=0).sum(1)
meta["cnv_del"] = (-AM.clip(max=0)).sum(1)
meta["n_arms_dev"] = (np.abs(AM) > 0.05).sum(1)
meta["cnv_call"] = meta["cnv_load"] > thr
meta["is_null_holdout"] = meta.index.isin(null_rows)

# ---- validation (normal cells are held out from reference construction) ----
from sklearn.metrics import roc_auc_score
def auroc(ds, a_lab):
    A = meta[(meta.dataset == ds) & (meta.compartment_clean == a_lab)].cnv_load.values
    B = meta.loc[NULLK.get(ds, np.array([], dtype=int)), "cnv_load"].values
    if len(A) < 20 or len(B) < 20: return np.nan, 0, 0
    return roc_auc_score(np.r_[np.ones(len(A)), np.zeros(len(B))], np.r_[A, B]), len(A), len(B)

rep = {"datasets": TARGETS, "device": "cuda (CuPy)", "W": W,
       "n_genes": int(n_genes), "thr_q99_matched_null": thr,
       "n_ref": {k: int(len(v)) for k, v in REF.items()},
       "n_null_holdout": {k: int(len(v)) for k, v in NULLK.items()},
       "null_median": float(np.median(null_load)), "null_q99": thr,
       "positive_controls": {}}
for ds in TARGETS:
    u, na, nb = auroc(ds, "malignant_epithelium")
    rep["positive_controls"][f"{ds}_malignant_epithelium_vs_heldout_normal_epi_AUROC"] = (
        None if np.isnan(u) else round(float(u), 4))
    rep["positive_controls"][f"{ds}_n"] = [na, nb]

scored_meta = meta.loc[selected_rows].copy()
def grouped_values(series):
    return {"|".join(map(str, key)): float(value) for key, value in series.items()}

rep["frac_called_overall"] = round(float(scored_meta.cnv_call.mean()), 4)
rep["frac_called_by_group"] = grouped_values(
    scored_meta.groupby(["dataset", "compartment_clean"]).cnv_call.mean().round(3))
rep["median_load_by_group"] = grouped_values(
    scored_meta.groupby(["dataset", "compartment_clean"]).cnv_load.median().round(4))

scored_meta.index = scored_meta.barcode.astype(str)
scored_meta[["dataset", "patient_id", "site", "stage", "compartment_clean",
             "cnv_load", "cnv_amp", "cnv_del", "n_arms_dev", "cnv_call",
             "is_null_holdout"]].to_csv(f"{OUT}/cnv_per_cell.csv")
with open(f"{OUT}/cnv_summary.json", "w") as f:
    json.dump(rep, f, indent=1)
print(json.dumps(rep, indent=1))
