#!/usr/bin/env python3
"""61_epithelium_novelty.py — 密度校正 + 类型限制的"异常上皮"判据（完全不依赖注释与 CNV）。

60 号脚本里裸的 novelty 不能用：它被参考图谱的类别密度差异支配
（IgA+Plasma 5.31 / myCAF 2.82 的"新奇度"比多数肿瘤 clone 还高）。
这里改成两件事：
  (1) 只在**参考的上皮/呼吸系统细胞**里找最近邻 -> 组内比较，密度差异天然被控住；
  (2) 再用"参考自身 1NN 距离"作零分布做密度校正：z = log d_query - log d_null。

关键检验（也是回答"那两个没注释的其实做过 CNV"）：
  GSE131907 的子图谱上皮细胞**全部**被 fine label 判成 Normal Alveolar Epithelial（0 个恶性）。
  同一供体的癌旁(LUNG_N##) vs 肿瘤(LUNG_T##) 两侧，用这个独立判据做配对检验：
  若肿瘤侧显著更远 -> 它们确实不是正常上皮，支持 CNV 的异常倍体结论，
                    即当前 fine label 在 GSE131907 上是错的（而不是"这个数据集没有肿瘤上皮"）。
"""
import os, sys, json
import numpy as np, pandas as pd, torch, faiss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L
from scipy import stats
from sklearn.metrics import roc_auc_score

RES = f"{L.RES}/model_native"
dev = "cuda"
import anndata as ad
A = ad.read_h5ad("/home/ubuntu/luad_scmg_v2/ref/data/ref_global_cell_state_manifold.h5ad", backed="r")
RefE = np.ascontiguousarray(np.asarray(A.obsm["X_scmg"][:]).astype(np.float32))
rmaj = np.asarray(A.obs["major_cell_type"].astype(object).fillna("unknown"), dtype=str)
rct = np.asarray(A.obs["cell_type"].astype(object).fillna("unknown"), dtype=str)

EPITH = {"Respiratory system", "Other epithelium", "Epithelial progenitor"}
sel = np.isin(rmaj, list(EPITH))
print("参考里的上皮细胞:", sel.sum(), "| 类型:", len(set(rct[sel])))
RefEpi = np.ascontiguousarray(RefE[sel])

# 零分布：参考上皮细胞自己互为最近邻（排除自身）
res = faiss.StandardGpuResources()


def knn(db, query, k):
    idx = faiss.IndexFlatL2(db.shape[1])
    g = faiss.index_cpu_to_gpu(res, 0, idx)
    g.add(np.ascontiguousarray(db))
    D, I = g.search(np.ascontiguousarray(query.astype(np.float32)), k)
    return D, I


d_null, _ = knn(RefEpi, RefEpi, 2)
null_med = float(np.median(d_null[:, 1]))            # 排除自身后的中位 1NN 距离
print(f"参考内 1NN 中位距离 d_null = {null_med:.3f}")

order = L.load_order()
ann = pd.read_csv(f"{L.RES}/cell_annotations_clean.csv", index_col=0)
nat = pd.read_csv(f"{RES}/cell_native_annotations.csv")
assert (nat.barcode.values == order.barcode.values).all()

epi_pos = np.where(np.isin(ann["compartment_clean"].values,
                           ["malignant_epithelium", "normal_alveolar_epithelial"]))[0]
Z = np.load("/tmp/luad_316k_scmg_embeddings.npy")
D_epi, _ = knn(RefEpi, Z[epi_pos], 1)
d_q = D_epi[:, 0]
sub = pd.DataFrame({
    "barcode": order["barcode"].values[epi_pos],
    "dataset": order["dataset"].values[epi_pos],
    "patient_id": order["patient_id"].values[epi_pos],
    "stage": order["stage"].values[epi_pos],
    "broad": ann["broad_cell_type"].values[epi_pos],
    "fine": ann["assigned_fine_label"].values[epi_pos],
    "compartment": ann["compartment_clean"].values[epi_pos],
    "native_cell_type": nat["native_cell_type"].values[epi_pos],
    "native_major": nat["native_major"].values[epi_pos],
    "d_to_normal_epithelium": d_q,
    "z_novelty": np.log(d_q + 1e-6) - np.log(null_med),
})
sub.to_csv(f"{RES}/epithelium_novelty.csv", index=False)
print("\n=== 各 dataset / 区室的『到正常上皮的距离』中位数 ===")
print(sub.groupby(["dataset", "compartment"])["d_to_normal_epithelium"].agg(["median", "count"]).to_string())

# ---------- 核心配对检验：GSE131907 同一供体 癌旁 vs 肿瘤 ----------
import re
g13 = sub[sub.dataset == "GSE131907"].copy()
g13["subj"] = [re.search(r"_(?:LUNG_N|LUNG_T)(\d+)$", p).group(1) if re.search(r"_(?:LUNG_N|LUNG_T)(\d+)$", p) else None
               for p in g13.patient_id]
g13["side"] = np.where(g13.patient_id.astype(str).str.contains("_LUNG_T"), "tumor",
                np.where(g13.patient_id.astype(str).str.contains("_LUNG_N"), "normal_lung", None))
pair = g13.dropna(subset=["subj", "side"])
ok = [s for s, gr in pair.groupby("subj") if (gr.side == "tumor").sum() >= 20 and (gr.side == "normal_lung").sum() >= 20]
print(f"\n=== GSE131907 同供体配对（{len(ok)} 个供体，上皮细胞全部被注释成 normal AT2）===")
diffs, rows = [], []
for s in ok:
    gr = pair[pair.subj == s]
    a = gr.loc[gr.side == "normal_lung", "d_to_normal_epithelium"]
    b = gr.loc[gr.side == "tumor", "d_to_normal_epithelium"]
    diffs.append(b.median() - a.median())
    rows.append(dict(subj=s, n_normal=len(a), n_tumor=len(b),
                     med_normal=round(float(a.median()), 3), med_tumor=round(float(b.median()), 3),
                     delta=round(float(b.median() - a.median()), 3)))
print(pd.DataFrame(rows).to_string(index=False))
diffs = np.array(diffs)
t_, p_ = stats.ttest_1samp(diffs, 0)
w_ = stats.wilcoxon(diffs)
print(f"\n配对差值均值 = {diffs.mean():.3f} | t={t_:.2f} p={p_:.3g} | Wilcoxon p={w_.pvalue:.3g} | "
      f"肿瘤侧更远的供体比例 = {(diffs > 0).mean():.2f}")
auc_all = roc_auc_score((pair.side == "tumor").astype(int), pair.d_to_normal_epithelium)
auc_within = np.mean([roc_auc_score((gr.side == "tumor").astype(int), gr.d_to_normal_epithelium)
                      for _, gr in pair.groupby("subj") if (gr.side == "tumor").sum() >= 20 and (gr.side == "normal_lung").sum() >= 20])
print(f"AUC(裸距离->哪一侧) 混合供体 = {auc_all:.3f} | 供体内平均 = {auc_within:.3f}")

# ---------- 与 native_cell_type 的交叉：被判成了什么 ----------
print("\n=== GSE131907 子图谱上皮的模型原生类型（GSE131907 没有任何恶性上皮注释）===")
print(g13[g13.broad == "Malignant"].native_cell_type.value_counts().head(12).to_string())
print("\n对照：GSE148071/189357 中被注释成恶性的细胞的 native 类型")
print(sub[sub.compartment == "malignant_epithelium"].native_cell_type.value_counts().head(12).to_string())

json.dump({"d_null_median": null_med, "n_paired_subjects": len(ok),
           "paired_delta_mean": round(float(diffs.mean()), 4), "paired_t_p": float(p_),
           "paired_wilcoxon_p": float(w_.pvalue), "frac_tumor_farther": round(float((diffs > 0).mean()), 3),
           "auc_pooled": round(float(auc_all), 4), "auc_within_subject_mean": round(float(auc_within), 4)},
          open(f"{RES}/epithelium_novelty.json", "w"), indent=2)
print("\nwrote", f"{RES}/epithelium_novelty.csv")
