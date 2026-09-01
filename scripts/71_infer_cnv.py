#!/usr/bin/env python3
"""71_infer_cnv.py -- expression-based CNV inference, implemented on GPU ourselves.

Why not run inferCNV: it needs R + dense matrices (~30 GB here), expects gene SYMBOLS
(our standard space is ENSG), and its usual workflow would have us define the normal
reference from the very annotation whose correctness is under test.

Statistic (inferCNV's core):
  1. GENCODE protein_coding genes on chr1-22,X, ordered by position;
  2. column-centre each gene by the median of the REFERENCE cells;
  3. clip each gene to +-CNV_CLIP * (its reference MAD), so a handful of outlier
     genes cannot dominate an arm mean;
  4. ONLY THEN row-centre each cell (subtract its genome-wide mean). This order is
     load-bearing -- see the note above the reference-statistics block;
  5. rolling mean over W consecutive genes, never crossing a chromosome;
  6. summarise to gene-bearing chromosome arms; score = mean |arm|.

Column-index convention (this is the bug that produced the invalid AUROC 0.395):
`cols` below is indexed in GENOMIC order (chr1 -> chrX, by start) because the sliding
window and the arm aggregation are only meaningful in that order. It is NOT the order
of `data/std_genes.npy`. Keeping both arrays around and picking the wrong one silently
averages genes that merely sit next to each other in the standard gene list, which
reduces the whole score to noise. `cols_std_order` exists only to build `cols`.

Honest null: the reference is HALF of the adjacent-normal normal-epithelium cells
(per dataset, split by index parity); the OTHER half is scored but never contributed to
the reference. Thresholds come from that held-out half, so the null is measured on cells
of the exact same biological state rather than on the reference cells themselves
(which would make the null artificially tight and the calls circular).

Positive control: GSE148071 is the ONLY usable one. Its 61 patients split cleanly into
P1-P5 (8,016 normal-alveolar cells, zero malignant) and P6-P61 (13,412 malignant
epithelial cells, zero normal). GSE189357 has 0 annotated normal epithelium and
GSE131907 has 0 annotated malignant epithelium, so neither can validate anything here.
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
CLIP = float(os.environ.get("CNV_CLIP", "3"))
EXPL = float(os.environ.get("CNV_EXPL", "0.10"))
MAD_FLOOR = float(os.environ.get("CNV_MAD_FLOOR", "0.05"))
CHROMS = [str(i) for i in range(1, 23)] + ["X"]
print(f"datasets: {', '.join(TARGETS)}\noutput: {OUT}", flush=True)

order = pd.read_csv(f"{L.DAT}/cell_order.csv")
ann = pd.read_csv(f"{L.RES}/cell_annotations_clean.csv", index_col=0)
pos = pd.read_csv(f"{L.RES}/gene_positions_gencode44.csv")
pos = pos[pos.gene_type == "protein_coding"].copy()
std = np.load(f"{L.DAT}/std_genes.npy", allow_pickle=True).astype(str)

# ---- gene space, positional order ----
pm = pos.set_index("ensg")
keep = [g for g in std if g in pm.index]
cols_std_order = np.array([np.where(std == g)[0][0] for g in keep])
gp = pm.loc[keep].reset_index(drop=True)
gp["chrom"] = gp.chrom.str[3:]
gp["col"] = cols_std_order
gp = gp[gp.chrom.isin(CHROMS)].copy()
gp["cnum"] = gp.chrom.map({c: i for i, c in enumerate(CHROMS)})
gp = gp.sort_values(["cnum", "start"]).reset_index(drop=True)

# Everything below is indexed in this genomic order.
cols = gp.col.values
CH = gp.chrom.values
ARMV = gp.arm.values
ARMS = sorted(set(ARMV))
n_genes = len(gp)
print(f"protein_coding genes on chr1-22,X: {n_genes}")
print(f"{len(ARMS)} chromosome arms")

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

# ---- reference gene medians, per-gene scale, expression filter ----
# Order matters and is easy to get backwards on sparse data. We subtract the reference
# profile FIRST (gene-wise), clip gene-wise, and only THEN remove each cell's own
# genome-wide mean. Doing it the other way round is a no-op here: more than half the
# genes are zero in any given 10x 3' cell, so a per-cell median taken on the raw log
# values is ~0 and leaves the library-size term in the score -- which then ranks cells
# by how much RNA they contain (immune 0.11 < malignant epi 0.20 < normal epi 0.23)
# instead of by copy number.
STAT = {}
for ds, idx in REF.items():
    B = X[np.ix_(idx, cols)].astype(cp.float32)
    expressed = np.where(cp.asnumpy((B > 0).mean(0)) >= EXPL)[0]
    B = B[:, expressed]
    med = cp.asnumpy(cp.median(B, 0)).astype(np.float32)      # reference level per gene
    # per-gene scale, used only to bound outliers (never to divide: that would amplify
    # noise on near-constant low-expressed genes, which are the majority here)
    mad = cp.asnumpy(cp.median(cp.abs(B - cp.array(med, dtype=cp.float32)[None, :]), 0))
    mad = np.maximum(mad.astype(np.float32), MAD_FLOOR)
    STAT[ds] = (expressed, med, mad)
    print(f"  {ds}: {len(expressed):,}/{n_genes:,} genes pass the {EXPL:.0%}-expressed filter")
    del B
print("reference gene medians done")


def arm_scores(rows, ds):
    expressed, med, mad = STAT[ds]
    use_cols = cols[expressed]
    local_ch = CH[expressed]
    arm_of = np.array([ARMS.index(a) for a in ARMV[expressed]])
    nA = len(ARMS)
    res = np.zeros((len(rows), nA), np.float32)
    cnt = np.bincount(arm_of, minlength=nA).astype(np.float32)
    bound = cp.array(CLIP * mad, dtype=cp.float32)
    for s in range(0, len(rows), 8000):
        b = rows[s:s + 8000]
        V = X[np.ix_(b, use_cols)].astype(cp.float32)
        V -= cp.array(med, dtype=cp.float32)[None, :]        # 1. remove reference level
        cp.clip(V, -bound, bound, out=V)                     # 2. bound outliers per gene
        V -= V.mean(1, keepdims=True)                        # 3. only now remove depth
        Sm = cp.zeros_like(V)
        for c in np.unique(local_ch):
            m = local_ch == c
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
del AMdf

# ---- per-patient consensus arm profiles ----
# A single cell at ~1,100 detected genes is far too sparse to call CNV on; the
# per-patient median is where arm-level events become visible, and it is what lets us
# ask whether malignant patients share RECURRENT events (batch effects do not recur
# on canonical LUAD arms, so this is a much harder test than an AUROC number).
EPI = {"malignant_epithelium", "normal_alveolar_epithelial"}


def consensus(mask, label):
    rows = meta.index[meta.dataset.isin(TARGETS) & mask].to_numpy()
    prof = pd.DataFrame(AM[rows], columns=ARMS)
    prof["dataset"] = meta.loc[rows, "dataset"].values
    prof["patient_id"] = meta.loc[rows, "patient_id"].values
    out = prof.groupby(["dataset", "patient_id"])[ARMS].median()
    out.insert(0, "n_cells", prof.groupby(["dataset", "patient_id"]).size())
    path = f"{OUT}/cnv_arm_by_patient{'_' + label if label else ''}.csv"
    out.to_csv(f"{path}.tmp")
    os.replace(f"{path}.tmp", path)
    print(f"  wrote {os.path.relpath(path, L.ROOT)}  ({len(out):,} patients)")
    return out


consensus(pd.Series(True, index=meta.index), "")
consensus(meta.compartment_clean.isin(EPI), "epithelium")

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

rep = {"datasets": TARGETS, "device": "cuda (CuPy)", "W": W, "clip": CLIP,
       "n_genes": int(n_genes),
       "n_genes_used": {ds: int(len(v[0])) for ds, v in STAT.items()},
       "thr_q99_matched_null": thr,
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
