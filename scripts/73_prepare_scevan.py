#!/usr/bin/env python3
"""Export raw UMI counts from the atlas h5ad for per-sample SCEVAN runs."""
import argparse
import gzip
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

import luadlib as L

H5AD = "/tmp/luad_316k_scmg_ready.h5ad"
DATASET = "GSE148071"

parser = argparse.ArgumentParser(description="Prepare sparse raw-count inputs for SCEVAN")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--patients", nargs="+")
group.add_argument("--all", action="store_true", help=f"Export every {DATASET} sample")
args = parser.parse_args()

adata = ad.read_h5ad(H5AD, backed="r")
order = L.load_order()
ann = pd.read_csv(f"{L.RES}/cell_annotations_clean.csv", index_col=0)
if len(order) != adata.n_obs or not np.array_equal(order.barcode.astype(str), adata.obs_names.astype(str)):
    raise RuntimeError("h5ad and cell_order rows are not aligned")

root = f"{L.RES}/scevan/input"
os.makedirs(root, exist_ok=True)
if args.all:
    patients = order.loc[order.dataset.eq(DATASET), "patient_id"].drop_duplicates().tolist()
    patients.sort(key=lambda value: int(str(value).rsplit("P", 1)[1]))
else:
    patients = args.patients
for patient in patients:
    rows = np.flatnonzero(order.dataset.eq(DATASET).to_numpy() & order.patient_id.eq(patient).to_numpy())
    if len(rows) == 0:
        raise ValueError(f"no cells found for {patient}")
    counts = sparse.csr_matrix(adata[rows].X)
    if counts.data.size and (counts.data.min() < 0 or not np.all(counts.data == np.round(counts.data))):
        raise ValueError(f"{patient} matrix is not non-negative integer raw counts")

    out = f"{root}/{patient}"
    os.makedirs(out, exist_ok=True)
    matrix_path = f"{out}/matrix.mtx.gz"
    with gzip.open(matrix_path, "wb") as handle:
        mmwrite(handle, counts.T.tocoo(), field="integer")
    pd.Series(adata.var_names.astype(str)).to_csv(
        f"{out}/genes.tsv", sep="\t", index=False, header=False)
    barcodes = order.iloc[rows].barcode.astype(str).to_numpy()
    pd.Series(barcodes).to_csv(f"{out}/barcodes.tsv", sep="\t", index=False, header=False)

    meta = order.iloc[rows].copy()
    clean = ann.reindex(barcodes)
    meta["compartment_clean"] = clean.compartment_clean.to_numpy()
    meta.to_csv(f"{out}/metadata.csv", index=False)
    manifest = {
        "source_h5ad": H5AD,
        "dataset": DATASET,
        "patient_id": patient,
        "shape_genes_by_cells": [int(counts.shape[1]), int(counts.shape[0])],
        "nnz": int(counts.nnz),
        "dtype": str(counts.dtype),
        "normal_labels_used_by_scevan": False,
    }
    with open(f"{out}/manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"{patient}: {counts.shape[0]:,} cells x {counts.shape[1]:,} genes, "
          f"{counts.nnz:,} nonzero counts -> {out}", flush=True)
