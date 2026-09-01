#!/usr/bin/env python3
"""60_model_native_annot.py — 用 SCMG 自带的全局映射给 316k 细胞重注释（不碰任何聚类标签）。

做法 = 官方 CellTypeSearcher 的语义：在**冻结的 zero-shot 空间**里，对全局参考图谱
(133,061 细胞 / 797 cell_type / 30 major_cell_type，含 11,501 个肺细胞) 做 kNN，
用 exp(-(d/radius)^2) 的软权重把参考细胞类型的隶属度迁移过来。

必须用冻结模型而不是微调后的模型：参考图谱的 X_scmg 是在 zero-shot 空间里算的，
换空间这张地图就失效（这也是"不该乱动 encoder"的直接理由）。

同时产出一个**与注释完全无关**的恶性判据：到全局参考的最近邻距离（novelty）。
参考里没有任何肿瘤数据集 -> 真正的恶性细胞应该"无处可归"、距离显著偏大。
"""
import os, sys, json
import numpy as np, pandas as pd, torch, faiss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

RES = f"{L.RES}/model_native"
os.makedirs(RES, exist_ok=True)
K = 25
RADIUS = 2.0
dev = "cuda"

ref = L.__dict__  # noqa
import anndata as ad
A = ad.read_h5ad("/home/ubuntu/luad_scmg_v2/ref/data/ref_global_cell_state_manifold.h5ad", backed="r")
RefE = np.ascontiguousarray(np.asarray(A.obsm["X_scmg"][:]).astype(np.float32))
rct = np.asarray(A.obs["cell_type"].astype(object).fillna("unknown"), dtype=str)
rmaj = np.asarray(A.obs["major_cell_type"].astype(object).fillna("unknown"), dtype=str)
rtis = np.asarray(A.obs["tissue"].astype(object).fillna("unknown"), dtype=str)
print("ref", RefE.shape, "cell_types", len(set(rct)))

q = np.load("/tmp/luad_316k_scmg_embeddings.npy")            # 冻结模型
order = L.load_order()
assert len(q) == len(order)

# 一致性自检：参考空间与我们的 embedding 是否同一尺度
print("ref norm mean %.2f | LUAD norm mean %.2f" %
      (np.linalg.norm(RefE, axis=1).mean(), np.linalg.norm(q, axis=1).mean()))

res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(RefE.shape[1])
gi = faiss.index_cpu_to_gpu(res, 0, index)
gi.add(RefE)
D, I = [], []
CH = 20000
for s in range(0, len(q), CH):
    d, i = gi.search(np.ascontiguousarray(q[s:s + CH].astype(np.float32)), K)
    D.append(d); I.append(i)
    print(f"  {s+CH}/{len(q)}", flush=True)
D = np.concatenate(D).astype(np.float32); I = np.concatenate(I).astype(np.int64)
print("kNN done", D.shape)

# ---- 软权重迁移（官方 radius 语义，按 class 计数归一；用 index_add_ 保证重复索引正确累加） ----
ct_uni, ct_inv = np.unique(rct, return_inverse=True)
maj_uni, maj_inv = np.unique(rmaj, return_inverse=True)
n_ct = np.bincount(ct_inv).astype(np.float32)
n_maj = np.bincount(maj_inv).astype(np.float32)
W = np.exp(-((D / RADIUS) ** 2))                     # (Nq, K)
rows = np.repeat(np.arange(len(q), dtype=np.int64), K)


def soft_scatter(inv, n_class, w):
    ncls = len(n_class)                          # 类别数
    idx = torch.as_tensor(rows * ncls + inv.reshape(-1), dtype=torch.long, device=dev)
    val = torch.as_tensor(w.reshape(-1) / n_class[inv.reshape(-1)], dtype=torch.float32, device=dev)
    M = torch.zeros(len(q) * ncls, dtype=torch.float32, device=dev)
    M.index_add_(0, idx, val)
    return M.view(len(q), ncls)


Sc_t = soft_scatter(ct_inv[I], n_ct, W)
Sm_t = soft_scatter(maj_inv[I], n_maj, W)
Sc_t /= (Sc_t.sum(1, keepdims=True) + 1e-12)
Sm_t /= (Sm_t.sum(1, keepdims=True) + 1e-12)
Sc = Sc_t.cpu().numpy()
maj_top = Sm_t.argmax(1).cpu().numpy()
maj_prob = Sm_t.max(1).values.cpu().numpy()
del Sc_t, Sm_t, W
torch.cuda.empty_cache()

top1 = Sc.argmax(1)
out = pd.DataFrame({
    "barcode": order["barcode"].values,
    "native_cell_type": ct_uni[top1],
    "native_prob": Sc.max(1).astype(np.float32),
    "native_major": maj_uni[maj_top],
    "native_major_prob": maj_prob.astype(np.float32),
    "native_entropy": (-(Sc * np.log(Sc + 1e-12)).sum(1)).astype(np.float32),
    "ref_dist_top1": D[:, 0].astype(np.float32),
    "ref_dist_meanK": D.mean(1).astype(np.float32),
})
for c in ["dataset", "patient_id", "stage", "malignant_call", "celltype_probe"]:
    out[c] = order[c].values
ann = pd.read_csv(f"{L.RES}/cell_annotations_clean.csv", index_col=0)
out["assigned_fine_label"] = ann["assigned_fine_label"].values
out["compartment_clean"] = ann["compartment_clean"].values
out["broad_cell_type"] = ann["broad_cell_type"].values
out.to_csv(f"{RES}/cell_native_annotations.csv", index=False)
print("\nnative_major 分布:\n" + out.native_major.value_counts().head(15).to_string())
print("\nnative_cell_type top15:\n" + out.native_cell_type.value_counts().head(15).to_string())

# ---- novelty：与注释无关的"恶性"信号 ----
from sklearn.metrics import roc_auc_score
print("\n=== 到全局参考的最近邻距离（越大=越无处可归）===")
g = out.groupby("assigned_fine_label")["ref_dist_top1"].agg(["median", "count"])
print(g.sort_values("median", ascending=False).head(12).to_string())
print("\n按区室:")
print(out.groupby("compartment_clean")["ref_dist_top1"].median().to_string())
print("\n按 dataset:")
print(out.groupby("dataset")["ref_dist_top1"].median().to_string())

auc = {}
for ds in out.dataset.unique():
    s = out[out.dataset == ds]
    m = s.compartment_clean.isin(["malignant_epithelium", "normal_alveolar_epithelial", "other"])
    a_ = s[m]
    if a_.compartment_clean.nunique() >= 2 and a_.groupby("compartment_clean").size().min() > 50:
        y = (a_.compartment_clean == "malignant_epithelium").astype(int).values
        if y.sum() and (~y.astype(bool)).sum():
            auc[f"AUC(novelty -> annotated-malignant) {ds}"] = round(float(roc_auc_score(y, a_.ref_dist_top1)), 4)
y = (out.compartment_clean == "malignant_epithelium").astype(int).values
auc["AUC(novelty -> annotated-malignant) ALL"] = round(float(roc_auc_score(y, out.ref_dist_top1)), 4)
yi = ((out.compartment_clean == "other") & out.broad_cell_type.isin(["T cell", "Macrophages"])).astype(int).values
auc["AUC(novelty vs immune/stroma sanity)"] = round(float(roc_auc_score(
    yi, -out.ref_dist_top1)), 4)
print("\n=== novelty 作为无标签恶性判据 ===")
print(json.dumps(auc, indent=1))

# ---- 与旧注释的一致性（证明管线本身没问题；分歧点应在上皮） ----
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
print("\n=== native_major vs 旧 broad_cell_type 的一致性 ===")
print("NMI = %.3f  ARI = %.3f" % (nmi(out.broad_cell_type, out.native_major),
                                  ari(out.broad_cell_type, out.native_major)))
ct = pd.crosstab(out.broad_cell_type, out.native_major)
print(ct.to_string())
epi = out[out.assigned_fine_label.astype(str).str.startswith("Tumor_") |
          (out.assigned_fine_label == "Normal Alveolar Epithelial")]
print("\n=== 旧上皮/肿瘤注释 -> 模型原生类型 (top) ===")
print(pd.crosstab(epi.assigned_fine_label, epi.native_major).to_string()[:2500])

json.dump({"K": K, "radius": RADIUS, "novelty_auc": auc,
           "nmi_native_vs_broad": float(nmi(out.broad_cell_type, out.native_major)),
           "ref_shape": list(RefE.shape)}, open(f"{RES}/annot_summary.json", "w"), indent=2)
print("\nwrote", RES)
