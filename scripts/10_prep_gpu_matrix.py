#!/usr/bin/env python3
"""10_prep_gpu_matrix.py — 一次性把 316,689 细胞转成 SCMG 标准 18,108 基因矩阵并落盘。

与官方 get_Xs_from_anndata 完全一致：zero-fill 到标准基因空间 -> CPRA 1e4 -> log1p。
存 float16（值域 <= ~9.3，fp16 精度足够），后续训练直接把整块矩阵搬进显存，
避免 /tmp 脚本里每个 epoch 重复 CPU->GPU 的开销。
"""
import os, sys, time, json
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
import pkg_resources

ROOT = "/home/ubuntu/luad_scmg_v2"
H5AD = "/tmp/luad_316k_scmg_ready.h5ad"
ANN = f"{ROOT}/results/cell_annotations_clean.csv"
DAT = f"{ROOT}/data"
os.makedirs(DAT, exist_ok=True)

std_genes = pd.read_csv(pkg_resources.resource_filename('scmg', 'data/standard_genes.csv'))['human_id'].values
NG = len(std_genes)

a = ad.read_h5ad(H5AD)
print("adata", a.shape, "X", type(a.X).__name__, a.X.dtype, flush=True)
assert a.isbacked is False

ann = pd.read_csv(ANN, index_col=0)
assert (ann.index == a.obs_names).all(), "注释表与 h5ad 行顺序不一致"
N, D = a.n_obs, NG

std_map = {g: i for i, g in enumerate(std_genes)}
adata_genes = np.asarray(a.var.index)
common = np.intersect1d(std_genes, adata_genes)
c_std = np.array([std_map[g] for g in common], dtype=np.int64)
c_adata = np.array([{g: i for i, g in enumerate(adata_genes)}[g] for g in common], dtype=np.int64)
print("common genes", len(common), flush=True)

X = a.X.tocsr()
mm = np.memmap(f"{DAT}/X_std_18108.f16", dtype=np.float16, mode="w+", shape=(N, D))
CH = 8192
t0 = time.time()
for s in range(0, N, CH):
    e = min(s + CH, N)
    blk = np.zeros((e - s, D), dtype=np.float32)
    blk[:, c_std] = X[s:e][:, c_adata].toarray().astype(np.float32)
    blk /= (blk.sum(1, keepdims=True) + 1e-6)
    blk *= 1e4
    np.log1p(blk, out=blk)
    mm[s:e] = blk.astype(np.float16)
    if (s // CH) % 10 == 0:
        print(f"  {e}/{N}  {time.time()-t0:.0f}s", flush=True)
mm.flush(); del mm, X, a
print(f"matrix done {time.time()-t0:.0f}s", flush=True)

# 顺序与元数据
order = pd.DataFrame({"barcode": np.asarray(ann.index),
                      "dataset": ann["dataset"].values,
                      "patient_id": ann["patient_id"].values,
                      "tissue_site": ann["tissue_site"].values,
                      "stage": ann["stage"].values,
                      "malignant_call": ann["malignant_call"].values,
                      "celltype_probe": ann["celltype_probe"].values})
order.to_csv(f"{DAT}/cell_order.csv", index=False)
np.save(f"{DAT}/std_genes.npy", np.asarray(std_genes, dtype="U16"))
json.dump({"n_cells": int(N), "n_genes": int(D), "common": int(len(common)),
           "dtype": "float16", "normalization": "CPM*1e4 (zero-filled to standard space) then log1p, "
           "identical to scmg.get_Xs_from_anndata(pre_normalized=False)"},
          open(f"{DAT}/prep_config.json", "w"), indent=2)
print("wrote", DAT, os.path.getsize(f"{DAT}/X_std_18108.f16") / 1e9, "GB")
