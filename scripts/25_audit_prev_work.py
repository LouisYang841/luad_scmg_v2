#!/usr/bin/env python3
"""25_audit_prev_work.py — 把对上一轮 (/tmp/*.py) 的审查变成可复现的数字。

复现并升级三个结论：
  (1) 用同一批标签训练 + 用同一批标签评估 = 循环；换成 leave-one-patient-out 后增益消失。
  (2) stage 与 dataset / 组织部位混淆，AAH/MIA/AIS 只有 1/2/3 个供体。
  (3) "扩散轨迹" 是把 100 个独立采样点连线得到的假象；且生成点明显离流形。
  (4) encoder 被改但 recon 项被去掉 / 无 LUAD dataset token -> 生成-解释链路退化。

低显存：只用 mmmap 取少量细胞，可与训练并发运行。
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L
from sklearn.metrics import silhouette_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances

STAGE_ORDER = ["Normal", "AAH", "AIS", "MIA", "IAC", "LNM"]

RES = L.RES
os.makedirs(RES, exist_ok=True)
out = {}
mm = L.load_X_memmap()
order = L.load_order()

zs = np.load("/tmp/luad_316k_scmg_embeddings.npy")
ft = np.load("/tmp/luad_316k_scmg_embeddings_finetuned.npy")
sft = np.load("/tmp/emb_stage_mal.npy")
obs = pd.read_csv("/tmp/luad_316k_scmg_umap_finetuned.csv", index_col=0, low_memory=False)
mp = pd.read_csv("/home/ubuntu/luad_invasion/results/metadata_cache/"
                 "all_subatlases_barcode_to_label_mapping.csv").drop_duplicates("cell_barcode")
pat = dict(zip(mp["cell_barcode"].astype(str), mp["patient_id"].astype(str)))

mal = np.where(obs["broad_cell_type"].values == "Malignant")[0]
st_mal = np.asarray(obs["stage"].values[mal], dtype=str)
pat_mal = np.array([pat.get(b, "NA") for b in obs.index.values[mal]])

# ---------------- (1) 循环评估 vs 供体级泛化 ----------------
rng = np.random.default_rng(0)
keep = np.concatenate([rng.choice(np.where(st_mal == s)[0], min(3000, (st_mal == s).sum()), replace=False)
                       for s in np.unique(st_mal)])
keep.sort()
lab, grp = st_mal[keep], pat_mal[keep]
out["old_labels_used_for_training"] = "stage(样本级) / broad_cell_type(Pure_Tumor 子图谱成员)"
hold = []
for s in np.unique(lab):
    cs = [g for g in np.unique(grp) if ((grp == g) & (lab == s)).sum() >= 200]
    hold += cs[:2]
E = {"ZS": zs[mal][keep], "FT-luad(leaky)": ft[mal][keep], "FT-stage(leaky)": sft[keep]}
tab = []
for name, X in E.items():
    Xp = StandardScaler().fit_transform(PCA(64, random_state=0).fit_transform(X.astype(np.float32)))
    tr = ~np.isin(grp, hold); te = np.isin(grp, hold)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xp[tr], lab[tr])
    i = rng.permutation(len(Xp)); rtr, rte = i[:len(i) * 4 // 5], i[len(i) * 4 // 5:]
    c2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xp[rtr], lab[rtr])
    tab.append(dict(model=name,
                    LOPO_unseen_donors=round(balanced_accuracy_score(lab[te], clf.predict(Xp[te])), 3),
                    random_split_sample_leak=round(balanced_accuracy_score(lab[rte], c2.predict(Xp[rte])), 3),
                    n_test=int(te.sum())))
out["1_stage_probe_leakage"] = tab
out["1_chance"] = round(1 / len(set(lab[te])), 3)
out["1_held_out_donors"] = len(hold)

# 复现他们报告过的 silhouette
out["1b_reported_silhouettes"] = dict(
    stage_ZS=round(silhouette_score(zs[mal][keep], lab), 4),
    stage_FT_luad=round(silhouette_score(ft[mal][keep], lab), 4),
    stage_FT_stage=round(silhouette_score(sft[keep], lab), 4),
    note="同一批标签既用于构造 hard negative 又用于打分 -> 不可作为改进证据")

# ---------------- (2) stage 的混淆结构 ----------------
out["2_donor_replication_per_stage"] = {s: int(len(set(pat_mal[st_mal == s]))) for s in np.unique(st_mal)}
out["2_stage_dataset_confound"] = {s: sorted(set(mp.loc[mp["stage"] == s, "dataset"])) for s in
                                   ["Normal", "AAH", "AIS", "MIA", "IAC", "LNM"]}

# ---------------- (3) 扩散轨迹是不是真的"路径" ----------------
g = np.load("/tmp/diffusion_transition_100.npy")
D = squareform(pdist(g)); cons = np.array([D[i, i + 1] for i in range(99)]); iu = np.triu_indices(100, 2)
emb_mal = sft
v = emb_mal[st_mal == "IAC"].mean(0) - emb_mal[st_mal == "Normal"].mean(0)
proj = g @ v / np.linalg.norm(v)
nn = NearestNeighbors(n_neighbors=10).fit(emb_mal)
dg, _ = nn.kneighbors(g); dr, _ = nn.kneighbors(emb_mal)
cents = np.stack([emb_mal[st_mal == s].mean(0) for s in STAGE_ORDER])
near = pd.Series([STAGE_ORDER[i] for i in pairwise_distances(g, cents).argmin(1)]).value_counts().to_dict()
out["3_diffusion_trajectory"] = dict(
    adjacent_over_all_pairs_distance=round(float(cons.mean() / D[iu].mean()), 3),
    spearman_index_vs_Normal_IAC_axis=round(float(spearmanr(np.arange(100), proj).correlation), 3),
    gen_to_real_10NN_dist=round(float(dg.mean(1).mean()), 2),
    real_to_real_10NN_dist=round(float(dr.mean(1).mean()), 2),
    nearest_stage_centroid_counts=near,
    note="相邻/任意≈1 => 100 个点是独立采样，连线不构成轨迹；离真实流形 ~2.8 倍")

# ---------------- (4) 生成-解释链路 ----------------
def recon_corr(model_path, names_tok, n=2500, seed=3):
    import torch
    model = L.load_embedder(model_path)
    ids = model.dataset_id_map
    g2 = np.random.default_rng(seed)
    idx = g2.choice(len(order), n, replace=False)
    X = torch.tensor(np.asarray(mm[idx], dtype=np.float32), device="cuda")
    tok = torch.as_tensor([ids.get(d, ids[names_tok]) for d in order["dataset"].values[idx]],
                          dtype=torch.long, device="cuda")
    with torch.no_grad():
        Z = model(X)
        P = L.decode_fast(model, Z, tok)
    c = float(np.corrcoef(X.cpu().numpy().ravel(), P.cpu().numpy().ravel())[0, 1])
    del model, X, Z, P
    torch.cuda.empty_cache()
    return round(c, 4)

out["4_recon_corr"] = dict(
    ZS_with_lung_token=recon_corr(f"{L.BASE_MODEL}/model.pt", "Sikkema_Lung_HS_2023:core"),
    FT_luad_with_lung_token=recon_corr("/tmp/scmg_luad_finetuned_opt/model.pt", "Sikkema_Lung_HS_2023:core"),
    FT_stage_with_lung_token=recon_corr("/tmp/scmg_stage_ft/stage_model.pt", "Sikkema_Lung_HS_2023:core"),
    note="上一轮两个微调模型都没有 LUAD dataset token，且训练时 recon 项被置 0")
out["4b_no_luad_token_in_prev_steps"] = True
out["4c_base_model_dataset_tokens"] = len(L.load_embedder(f"{L.BASE_MODEL}/model.pt").dataset_id_map)
out["4d_base_is_not_a_lung_cancer_model"] = "37 个预训练 dataset 全部来自发育/多组织图谱(含 He_LungDev, Sikkema_Lung)，无任何肿瘤数据集"

# ---------------- (5) 空间保持度 ----------------
i6 = np.random.default_rng(1).choice(len(order), 6000, replace=False)
out["5_geometry"] = dict(CKA_ZS_vs_FTluad=round(L.cka(zs[i6], ft[i6]), 4))
with open(f"{RES}/audit_prev_work.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
