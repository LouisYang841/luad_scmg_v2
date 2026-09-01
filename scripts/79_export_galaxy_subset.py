#!/usr/bin/env python3
"""79_export_galaxy_subset.py -- export the GSE148071 epithelium subset for inferCNV.

One combined 10x triplet covering every GSE148071 epithelial cell
(8,016 normal alveolar from P1-P5 + 13,412 malignant from P6-P61), plus a metadata
table marking which cells form the inferCNV reference (P1-P5 normal epithelium).
Raw counts are read from the atlas h5ad in backed mode, so memory stays flat.
Output: results/galaxy_infercnv/input/{matrix.mtx.gz, genes.tsv, barcodes.tsv, metadata.csv}
"""
import gzip
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

H5AD = "/tmp/luad_316k_scmg_ready.h5ad"
DATASET = "GSE148071"
EPI = {"malignant_epithelium", "normal_alveolar_epithelial"}

out_root = f"{L.RES}/galaxy_infercnv/input"
os.makedirs(out_root, exist_ok=True)

adata = ad.read_h5ad(H5AD, backed="r")
order = L.load_order()
ann = pd.read_csv(f"{L.RES}/cell_annotations_clean.csv", index_col=0)
if len(order) != adata.n_obs or not np.array_equal(order.barcode.astype(str), adata.obs_names.astype(str)):
    raise RuntimeError("h5ad and cell_order rows are not aligned")

mask_ds = order.dataset.eq(DATASET).to_numpy()
mask_epi = ann.reindex(order.barcode.astype(str))["compartment_clean"].isin(EPI).to_numpy()
rows = np.flatnonzero(mask_ds & mask_epi)
print(f"{DATASET} epithelium: {len(rows):,} cells", flush=True)

comp = ann.reindex(order.barcode.astype(str)).iloc[rows]["compartment_clean"].to_numpy()
pat = order.iloc[rows]["patient_id"].to_numpy()
is_ref = (comp == "normal_alveolar_epithelial") & pd.Series(pat).str.fullmatch(r"(?:.*_)?P[1-5]").to_numpy()
print(f"  malignant: {(comp == 'malignant_epithelium').sum():,}  "
      f"normal: {(comp == 'normal_alveolar_epithelial').sum():,}  "
      f"reference (P1-P5 normal): {is_ref.sum():,}", flush=True)

counts = sparse.csr_matrix(adata[rows].X)
if counts.data.size and (counts.data.min() < 0 or not np.all(counts.data == np.round(counts.data))):
    raise ValueError("matrix is not non-negative integer raw counts")

matrix_path = f"{out_root}/matrix.mtx.gz"
with gzip.open(matrix_path, "wb") as handle:
    mmwrite(handle, counts.T.tocoo(), field="integer")
pd.Series(adata.var_names.astype(str)).to_csv(f"{out_root}/genes.tsv", sep="\t", index=False, header=False)
pd.Series(order.iloc[rows].barcode.astype(str).to_numpy()).to_csv(
    f"{out_root}/barcodes.tsv", sep="\t", index=False, header=False)

meta = order.iloc[rows][["barcode", "dataset", "patient_id"]].astype(str).copy()
meta["compartment_clean"] = comp
meta["is_reference"] = is_ref
meta.to_csv(f"{out_root}/metadata.csv", index=False)

manifest = {
    "source_h5ad": H5AD,
    "dataset": DATASET,
    "n_cells": int(counts.shape[0]),
    "n_genes": int(counts.shape[1]),
    "nnz": int(counts.nnz),
    "n_reference": int(is_ref.sum()),
    "reference_definition": "P1-P5 normal_alveolar_epithelial",
    "gene_id_type": "ENSG (same space as results/gene_positions_gencode44.csv)",
}
with open(f"{out_root}/manifest.json", "w") as handle:
    json.dump(manifest, handle, indent=2)
print(f"wrote {out_root}  ({counts.shape[0]:,} cells x {counts.shape[1]:,} genes, {counts.nnz:,} nnz)")
