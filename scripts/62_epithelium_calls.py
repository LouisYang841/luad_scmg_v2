#!/usr/bin/env python3
"""62_epithelium_calls.py — 用参考图谱自身的"内距离"作零分布，给出与注释/CNV 无关的上皮异常判定。

判据：一个上皮细胞到参考中**正常呼吸/肺上皮**的最近邻距离 d_query。
零分布 = 参考上皮细胞互相之间的 1NN 距离 d_null（即"一个真实正常上皮细胞
离它最近的同类有多远"）。取 d_null 的 99/99.9 分位为阈值：
  d_query > q99(d_null)  =>  这个细胞离正常上皮流形比 99% 的真实正常上皮细胞都远
                          =>  判为 abnormal_epithelium（不是正常上皮）
这条阈值与被比较的 dataset 无关，也不看任何人工注释。
"""
import os, sys, json
import numpy as np, pandas as pd, faiss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

RES = f"{L.RES}/model_native"
import anndata as ad
A = ad.read_h5ad("/home/ubuntu/luad_scmg_v2/ref/data/ref_global_cell_state_manifold.h5ad", backed="r")
RefE = np.ascontiguousarray(np.asarray(A.obsm["X_scmg"][:]).astype(np.float32))
rmaj = np.asarray(A.obs["major_cell_type"].astype(object).fillna("unknown"), dtype=str)
rct = np.asarray(A.obs["cell_type"].astype(object).fillna("unknown"), dtype=str)
EPITH = {"Respiratory system", "Other epithelium", "Epithelial progenitor"}
RefEpi = np.ascontiguousarray(RefE[np.isin(rmaj, list(EPITH))])
# 只保留肺相关的参考细胞做零分布会更严，但样本量小；两者都算
RefLung = np.ascontiguousarray(RefE[np.isin(rmaj, ["Respiratory system"])])

res = faiss.StandardGpuResources()


def self_knn(db, k=2):
    idx = faiss.IndexFlatL2(db.shape[1])
    g = faiss.index_cpu_to_gpu(res, 0, idx)
    g.add(db)
    D, _ = g.search(db, k)
    return D[:, 1]


d_null = self_knn(RefEpi)
d_null_lung = self_knn(RefLung)
q = lambda v, p: float(np.quantile(v, p))
thr = {"epi_q99": q(d_null, .99), "epi_q999": q(d_null, .999), "epi_q95": q(d_null, .95),
       "lung_q99": q(d_null_lung, .99), "lung_q999": q(d_null_lung, .999)}
print("参考上皮内 1NN 距离: median %.3f | " % np.median(d_null), json.dumps({k: round(v, 3) for k, v in thr.items()}))
print("参考 lung-only 内 1NN: median %.3f (n=%d)" % (np.median(d_null_lung), len(d_null_lung)))

s = pd.read_csv(f"{RES}/epithelium_novelty.csv")
s["abnormal_q99"] = (s.d_to_normal_epithelium > thr["epi_q99"]).astype(int)
s["abnormal_q999"] = (s.d_to_normal_epithelium > thr["epi_q999"]).astype(int)
import re
side = np.where(s.patient_id.astype(str).str.contains("_LUNG_T"), "tumor_sample",
        np.where(s.patient_id.astype(str).str.contains("_LUNG_N"), "adjacent_normal",
        np.where(s.patient_id.astype(str).str.contains("_LN_|_EBUS_"), "lymph_node",
        np.where(s.patient_id.astype(str).str.contains("EFFUSION"), "pleural_effusion", "other"))))
s["site"] = np.where(s.dataset == "GSE131907", side, s.stage)
pd.set_option("display.width", 200)
print("\n=== 被判为『非正常上皮』的比例（阈值=参考内 99 分位）===")
print(s.groupby(["dataset", "compartment"])
      .agg(med_dist=("d_to_normal_epithelium", "median"),
           abnormal99=("abnormal_q99", "mean"), abnormal999=("abnormal_q999", "mean"),
           n=("abnormal_q99", "size")).round(3).to_string())
print("\n=== GSE131907（原注释里 0 个恶性上皮）按采集部位 ===")
print(s[s.dataset == "GSE131907"].groupby("site")
      .agg(med_dist=("d_to_normal_epithelium", "median"), abnormal99=("abnormal_q99", "mean"),
           abnormal999=("abnormal_q999", "mean"), n=("abnormal_q99", "size")).round(3).to_string())
print("\n=== 与 stage 的关系（跨 dataset，注意 stage 本身与 dataset 混淆）===")
print(s.groupby("stage").agg(med_dist=("d_to_normal_epithelium", "median"),
                             abnormal99=("abnormal_q99", "mean"), n=("abnormal_q99", "size")).round(3).to_string())

# 每个供体的判定，便于后面做供体级统计
don = s.groupby(["dataset", "patient_id", "site"]).agg(
    n_epi=("abnormal_q99", "size"), frac_abnormal=("abnormal_q99", "mean"),
    med_dist=("d_to_normal_epithelium", "median")).reset_index()
don.to_csv(f"{RES}/donor_epithelium_calls.csv", index=False)
print("\n=== 供体级（GSE131907 癌旁 vs 肿瘤，9 对）===")
d13 = don[don.dataset == "GSE131907"].dropna(subset=["site"])
print(d13[d13.site.isin(["adjacent_normal", "tumor_sample"])].sort_values(["patient_id"]).to_string(index=False))
s.to_csv(f"{RES}/epithelium_calls.csv", index=False)
json.dump({"thresholds": thr, "n_ref_epi": int(len(RefEpi)), "n_ref_lung": int(len(RefLung))},
          open(f"{RES}/epithelium_thresholds.json", "w"), indent=2)
print("\nwrote", f"{RES}/epithelium_calls.csv", "|", f"{RES}/donor_epithelium_calls.csv")
