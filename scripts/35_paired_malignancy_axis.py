#!/usr/bin/env python3
"""35_paired_malignancy_axis.py — 用"供体级 / 同供体配对"取代上一轮的跨样本 stage 对比。

上一轮的 Normal→IAC latent direction 是拿两个不同 dataset、不同组织部位、不同供体的组做差，
差出来的向量里混着供体效应、解离效应、dataset 效应，且把细胞当独立重复。

本脚本给出两个能真正立住的设计：

  A) 主分析（不配对，但推断单元 = 供体）
     GSE148071：56 个肿瘤供体(有 malignant 上皮) vs 5 个正常肺供体(只有 AT2 上皮)。
     embedding 侧用 leave-one-donor-out 分类器（见 30/37），基因侧用供体 pseudobulk 的两样本检验。

  B) 敏感性分析（真正的同供体配对）
     GSE131907 的 LUNG_T## / LUNG_N## 编号有 10 个重合，按命名约定推断为**同一供体的肿瘤与癌旁**。
     对该 10 个供体取"癌旁上皮 vs 肿瘤内上皮"的供体内差向量 u_d：供体效应被完全消掉。
     跨供体比较 u_d 的方向一致性（mean resultant length R）+ LOPO 可迁移准确率。
     注意：GSE131907 的上皮全部被注释成 normal AT2（0 个 malignant），所以 B 检验的是
     "AT2 程序在肿瘤微环境里的改变"，不是"是否恶性"。

     ⚠ 配对关系由样本名推断，未与 GEO characteristics 核对；若约定不成立，B 自动退化。
"""
import os, sys, json, re
import numpy as np, pandas as pd, torch
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RES, DAT = L.RES, L.DAT
order = L.load_order()
STD = np.load(f"{DAT}/std_genes.npy")
gn = pd.read_csv("/home/ubuntu/scmg_workspace/SCMG/scmg/data/standard_genes.csv")
id2name = dict(zip(gn["human_id"], gn["human_name"]))
dsn = sorted(order["dataset"].unique())
donor = order["patient_id"].values
mal_m = (order["malignant_call"] == "malignant").values
at2_m = (order["malignant_call"] == "normal_at2").values

# ---------- 可识别性：哪些配对设计真的存在 ----------
print("=== 供体级可识别性 ===")
same_donor_both = [d for d in np.unique(donor) if mal_m[donor == d].sum() >= 30 and at2_m[donor == d].sum() >= 30]
print(f"同一 patient_id 内同时有 malignant 与 normal_at2 的供体: {len(same_donor_both)}  -> 该设计不可用")


def subj131907(pid):
    m = re.search(r"_(LUNG_[NT])(\d+)$", pid)
    if not m:
        return None, None
    site, num = m.groups()
    return f"SUBJ{num}", ("tumor" if site.endswith("T") else "normal")


skey, stype = np.full(len(order), None, dtype=object), np.full(len(order), None, dtype=object)
for i, p in enumerate(donor):
    if order["dataset"].values[i] == "GSE131907":
        skey[i], stype[i] = subj131907(p)
cand_keys = sorted(set(k for k in skey if k is not None))
ok_subj = []
for k in cand_keys:
    i = np.where(skey == k)[0]
    if ((at2_m[i] & (stype[i] == "tumor")).sum() >= 30
            and (at2_m[i] & (stype[i] == "normal")).sum() >= 30):
        ok_subj.append(k)
print(f"GSE131907 推断的癌旁/肿瘤同供体配对(且两侧>=30 个 AT2 细胞): {len(ok_subj)}  ⚠配对由样本名推断")
g148 = order["dataset"].values == "GSE148071"
d_mal = sorted(set(donor[g148 & mal_m])); d_at2 = sorted(set(donor[g148 & at2_m]))
print(f"GSE148071 供体级两样本设计: {len(d_mal)} 个肿瘤供体 vs {len(d_at2)} 个正常肺供体（主分析）")

# ---------- embedding ----------
SP = {"ZS": np.load("/tmp/luad_316k_scmg_embeddings.npy"),
      "FT-luad(leaky)": np.load("/tmp/luad_316k_scmg_embeddings_finetuned.npy")}
for cand in ["chosen_state.pth", "best_state_dict.pth"]:
    f = f"{RES}/finetune_unsup/{cand}"
    if os.path.exists(f):
        m = L.load_any(f, extra_token_names=dsn)
        m, ids, _ = L.add_dataset_tokens(m, dsn)
        X = L.x_to_gpu()
        with torch.no_grad():
            e = np.concatenate([m(X[s:s + 16384].float()).float().cpu().numpy()
                                for s in range(0, len(order), 16384)])
        SP["FT-unsup"] = e
        del m, X, e
        torch.cuda.empty_cache()
        break

# ---------- B) 同供体配对恶性轴 ----------
rows = {}
for tag, emb in SP.items():
    U = []
    for k in ok_subj:
        i = np.where(skey == k)[0]
        a = i[(stype[i] == "normal") & at2_m[i]]     # 癌旁 AT2
        b = i[(stype[i] == "tumor") & at2_m[i]]      # 同供体肿瘤内 AT2
        if len(a) >= 30 and len(b) >= 30:
            U.append(emb[b].mean(0) - emb[a].mean(0))
    if len(U) < 3:
        rows[tag] = dict(n_paired_subjects=len(U)); continue
    U = np.stack(U)
    v = U.mean(0); vh = v / np.linalg.norm(v)
    cos = U @ vh / np.linalg.norm(U, axis=1)
    acc = []
    for j in range(len(U)):
        w = np.delete(U, j, 0).mean(0); w /= np.linalg.norm(w)
        k = ok_subj[j]; i = np.where(skey == k)[0]
        a = i[(stype[i] == "normal") & at2_m[i]]; b = i[(stype[i] == "tumor") & at2_m[i]]
        # 供体层面：用其余供体学到的轴，这个未见供体的肿瘤侧均值是否排在癌旁侧之上
        acc.append(float((emb[b] @ w).mean() > (emb[a] @ w).mean()))
    rows[tag] = dict(n_paired_subjects=len(U), mean_resultant_R=round(float(cos.mean()), 3),
                     cos_min=round(float(cos.min()), 3), LOPO_subject_rank_acc=round(float(np.mean(acc)), 3),
                     axis_norm=round(float(np.linalg.norm(U, axis=1).mean()), 3))
print("\n=== B) 同供体配对轴（癌旁AT2 -> 肿瘤内AT2，消掉供体效应）===")
print(pd.DataFrame(rows).T.to_string())

# ---------- A) 供体级两样本 DE（GSE148071，推断单元=供体）----------
mm = L.load_X_memmap()


def donor_pseudo(donors_list, mask):
    """供体级平均表达谱。注：X 已是 log1p(CPM*1e4)，所以这里取的是
    “细胞平均 log 表达”（不是累加计数的 pseudobulk）；推断单元仍是供体。"""
    out = {}
    for d in donors_list:
        idx = np.where((donor == d) & mask)[0]
        if len(idx) < 20:
            continue
        acc = np.zeros(len(STD))
        for s in range(0, len(idx), 4096):
            acc += np.asarray(mm[idx[s:s + 4096]], dtype=np.float32).sum(0)
        out[d] = acc / len(idx)
    return pd.DataFrame(out).T


A = donor_pseudo(d_mal, mal_m)
B = donor_pseudo(d_at2, at2_m)
print(f"\n=== A) 供体级 pseudobulk：肿瘤 {A.shape} vs 正常肺 {B.shape} ===")
common = np.arange(len(STD))
D = np.stack([A.values.mean(0) - B.values.mean(0)])   # (1, genes) 供体均值之差
tt, pp = stats.ttest_ind(A.values, B.values, axis=0, equal_var=False)
mw, pm = stats.mannwhitneyu(A.values, B.values, axis=0)
res = pd.DataFrame({"ensg": STD, "symbol": [id2name.get(g, g) for g in STD],
                    "mean_logFC": D[0], "welch_t": tt, "p_welch": pp, "p_mwu": pm,
                    "n_tumor_donors": A.shape[0], "n_normal_donors": B.shape[0]})
res = res.replace([np.inf, -np.inf], np.nan)
try:
    from statsmodels.stats.multitest import multipletests
    ok = res["p_welch"].notna()
    res.loc[ok, "q_welch"] = multipletests(res.loc[ok, "p_welch"], method="fdr_bh")[1]
except Exception:
    res["q_welch"] = np.minimum(1, res["p_welch"] * len(res))
res.to_csv(f"{RES}/donor_level_de_GSE148071.csv", index=False)
print("\n上调 top12（肿瘤供体 - 正常供体）:")
print(res.sort_values("welch_t", ascending=False).head(12)[["symbol", "mean_logFC", "welch_t", "p_welch", "q_welch"]].round(4).to_string(index=False))
print("\n下调 top12:")
print(res.sort_values("welch_t").head(12)[["symbol", "mean_logFC", "welch_t", "p_welch", "q_welch"]].round(4).to_string(index=False))
mk = [g for g in ["SFTPC", "SFTPB", "NAPSA", "SFTPA1", "LAMP3", "PUM2", "EPCAM", "KRT19", "MKI67", "TOP2A", "CLDN4", "SPP1"] if g in set(res["symbol"])]
print("\nmarker 对照:")
print(res[res["symbol"].isin(mk)][["symbol", "mean_logFC", "welch_t", "p_welch", "q_welch"]].sort_values("welch_t", ascending=False).round(4).to_string(index=False))

json.dump({"identifiability": dict(same_donor_paired=0, gse131907_inferred_pairs=len(ok_subj),
                                   gse148071_tumor_donors=len(d_mal), gse148071_normal_donors=len(d_at2),
                                   pairing_caveat="GSE131907 的 T##/N## 同供体配对由样本名推断，未核对 GEO characteristics"),
             "B_paired_axis": rows}, open(f"{RES}/paired_axis.json", "w"), indent=2, default=str)
print("\nwrote", f"{RES}/paired_axis.json")
