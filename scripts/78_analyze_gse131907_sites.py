#!/usr/bin/env python3
"""78_analyze_gse131907_sites.py -- where do the "normal alveolar" CNV calls land?

The hypothesis (plan 2026-09-01, finding 4): most of GSE131907's 23,350 cells labelled
normal_alveolar_epithelial are NOT from normal lung -- thousands sit in lymph nodes or
pleural effusion, i.e. metastatic sites, and must be malignant. 71_infer_cnv.py scored
every cell with a reference built ONLY from adjacent-normal (LUNG_N) epithelium, so if
the hypothesis is right the per-cell CNV load of the mislabelled cells should approach
the malignant level, stratifying cleanly by tissue_site.

Read-only: consumes results/cnv/GSE131907/cnv_per_cell.csv + cell_annotations_clean.csv.
Also splits the adjacent-normal reference supply by site as a sanity check, and prints
the per-site call-rate table that REPORT section 10 will quote.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

OUT = f"{L.RES}/cnv/GSE131907"
per_cell = pd.read_csv(f"{OUT}/cnv_per_cell.csv", index_col=0)
ann = pd.read_csv(f"{L.RES}/cell_annotations_clean.csv", index_col=0)

pc = per_cell.copy()
pc["compartment_clean"] = ann.reindex(pc.index)["compartment_clean"].values
pc["tissue_site"] = ann.reindex(pc.index)["tissue_site"].fillna("NA").values
pc["patient_id"] = ann.reindex(pc.index)["patient_id"].values

epi = pc[pc.compartment_clean == "normal_alveolar_epithelial"].copy()
thr = json.load(open(f"{OUT}/cnv_summary.json"))["thr_q99_matched_null"]
epi["called"] = epi["cnv_load"] > thr

print(f"threshold (q99 of matched-state held-out null): {thr:.4f}")
print(f"normal_alveolar_epithelial cells: {len(epi):,}\n")

tab = epi.groupby("tissue_site").agg(
    n_cells=("cnv_load", "size"),
    median_load=("cnv_load", "median"),
    frac_called=("called", "mean"),
).sort_values("n_cells", ascending=False)
print("=== call rate by tissue_site (normal_alveolar_epithelial) ===")
print(tab.to_string(float_format=lambda v: f"{v:.3f}"))

# patient-level view for the two sites that matter most
for site in ["normal_lung", "lymph_node", "pleural_effusion", "primary_lung_tumor"]:
    sub = epi[epi.tissue_site == site]
    if len(sub) == 0:
        continue
    byp = sub.groupby("patient_id")["called"].agg(["size", "mean"])
    hi = (byp["mean"] > 0.5).sum()
    print(f"\n{site}: {len(sub):,} cells in {len(byp)} patients; "
          f"{hi} patients with >50% cells called; overall {sub.called.mean():.1%} called")

# context: what does the rest of the tissue look like at the same sites?
print("\n=== context: cnv_load by compartment x tissue_site (medians) ===")
ctx = pc.pivot_table(index="tissue_site", columns="compartment_clean",
                     values="cnv_load", aggfunc="median")
print(ctx.to_string(float_format=lambda v: f"{v:.4f}"))

out_path = f"{OUT}/cnv_calls_by_site.csv"
tab.to_csv(out_path)
print(f"\nwrote {os.path.relpath(out_path, L.ROOT)}")
