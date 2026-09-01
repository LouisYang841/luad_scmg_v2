#!/usr/bin/env python3
"""37_marker_benchmark.py — 与注释半独立的留出供体基准。

上一轮所有指标都建立在同一份可疑注释上。这里用标志基因面板构造**不依赖聚类注释**的信号，
并在两个真实存在的设计上测 leave-one-donor-out 可迁移性：

  (i)  GSE148071 跨供体：annotated malignant 上皮 vs annotated normal AT2
       （label 用注释；marker 面板只作为参照上界，不参与标签）
  (ii) GSE131907 同供体内配对：只取被注释成 normal AT2 的上皮细胞，
       label = 该细胞来自"肿瘤样本"还是"癌旁正常样本"（同一供体两侧都有）。
       这个 label 完全不用 Tumor_C* 聚类注释，且供体效应可以消掉；
       marker 面板的供体内配对得分作为独立参照与一致性检验。

AUROC 越高 = 该空间携带的信号越能被**未见供体**读出来。
marker 参照 = 原始表达里确实存在的信号强度上界；embedding 若低于它，说明在丢信息。
"""
import os, sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

RES, DAT = L.RES, L.DAT
STD = np.load(f"{DAT}/std_genes.npy")
gn = pd.read_csv("/home/ubuntu/scmg_workspace/SCMG/scmg/data/standard_genes.csv")
sym2idx = {}
for s, e in zip(gn["human_name"], gn["human_id"]):
    w = np.where(STD == e)[0]
    if len(w):
        sym2idx.setdefault(s, int(w[0]))
AT2 = [g for g in ["SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "NAPSA", "LAMP3", "SFTPD", "PGC",
                   "CTSH", "FOLR1", "SLC34A2", "NKX2-1", "ETV5", "AKR1B1", "LIPF"] if g in sym2idx]
TU = [g for g in ["KRT19", "EPCAM", "TMPRSS2", "CEACAM6", "LGALS4", "MUC5B", "SPP1", "KRT5",
                  "TP63", "ALDH2", "SERPINB3", "CLDN4", "AREG", "HBEGF", "VMAC", "TIMP1"] if g in sym2idx]
print("AT2 panel   :", AT2)
print("气道/肿瘤panel:", TU)

order = L.load_order()
donor = order["patient_id"].values
ds = order["dataset"].values
call_ann = (order["malignant_call"] == "malignant").values
call_at2 = (order["malignant_call"] == "normal_at2").values

# ---------- 显存里算 marker 得分（只需一次） ----------
X = L.x_to_gpu()
ia = torch.as_tensor([sym2idx[g] for g in AT2], device="cuda")
ib = torch.as_tensor([sym2idx[g] for g in TU], device="cuda")
with torch.no_grad():
    sc_list = []
    for s in range(0, len(order), 8192):                 # 分块，避免 316k×18108 的 fp32 拷贝爆显存
        blk = X[s:s + 8192].float()
        sc_list.append((blk[:, ib].mean(1) - blk[:, ia].mean(1)).cpu())
        del blk
    score_raw = torch.cat(sc_list).numpy()

# (ii) 用的供体内配对得分
import re
skey = np.array([("SUBJ" + re.search(r"_(LUNG_[NT])(\d+)$", p).group(2))
                 if (ds[i] == "GSE131907" and re.search(r"_(LUNG_[NT])(\d+)$", p)) else None
                 for i, p in enumerate(donor)], dtype=object)
stype = np.array([("tumor" if "_LUNG_T" in p else "normal")
                  if (ds[i] == "GSE131907" and re.search(r"_(LUNG_[NT])\d+$", p)) else None
                  for i, p in enumerate(donor)], dtype=object)


def donor_paired(s, keys):
    out = np.full(len(s), np.nan)
    for k in keys:
        i = np.where(skey == k)[0]
        out[i] = (s[i] - np.nanmean(s[i])) / (np.nanstd(s[i]) + 1e-6)
    return out


# ---------- embedding 空间 ----------
SP = {"ZS": np.load("/tmp/luad_316k_scmg_embeddings.npy"),
      "FT-luad(leaky)": np.load("/tmp/luad_316k_scmg_embeddings_finetuned.npy")}
for cand in ["chosen_state.pth", "best_state_dict.pth"]:
    f = f"{RES}/finetune_unsup/{cand}"
    if os.path.exists(f):
        m = L.load_any(f, extra_token_names=sorted(pd.unique(ds)))
        with torch.no_grad():
            e = np.concatenate([m(X[s:s + 8192].float()).float().cpu().numpy()
                                for s in range(0, len(order), 8192)])
        SP["FT-unsup"] = e
        del m, e
        torch.cuda.empty_cache()
        break


def lodo_auc(E, y, groups, n_splits=6, pcs=48):
    k = min(pcs, E.shape[1], max(2, E.shape[0] - 1))
    Ep = PCA(k, random_state=0).fit_transform(E.astype(np.float32)) if k < E.shape[1] else E.astype(np.float32)
    Ep = StandardScaler().fit_transform(Ep)
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Ep, y, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ep[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Ep[te])[:, 1]))
    return (round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4), len(aucs), int(len(y)))


results = {}

# ============ (i) GSE148071 跨供体：malignant vs normal AT2 ============
i1 = np.where((ds == "GSE148071") & (call_ann | call_at2))[0]
y1 = call_ann[i1].astype(int)
g1 = donor[i1]
print(f"\n=== (i) GSE148071 跨供体 恶性 vs 正常AT2 : n={len(i1)} cells, {len(np.unique(g1))} donors "
      f"(对照供体只有 {len(set(g1[y1==0]))} 个 -> 功效有限)) ===")
tab = []
ref = np.column_stack([score_raw[i1], -score_raw[i1]])
a0 = lodo_auc(ref, y1, g1, n_splits=6)
tab.append(dict(space="[ref] marker panel", LODO_AUROC=a0[0], sd=a0[1], folds=a0[2]))
for tag, E in SP.items():
    tab.append(dict(space=tag, **dict(zip(("LODO_AUROC", "sd", "folds"), lodo_auc(E[i1], y1, g1)[:3]))))
results["i_GSE148071_cross_donor"] = tab
print(pd.DataFrame(tab).sort_values("LODO_AUROC", ascending=False).to_string(index=False))

# ============ (ii) GSE131907 同供体内：癌旁 vs 肿瘤（只用 AT2 注释细胞） ============
subj = []
for k in sorted(set(x for x in skey if x)):
    sel = skey == k
    if ((sel & (stype == "tumor") & call_at2).sum() >= 30
            and (sel & (stype == "normal") & call_at2).sum() >= 30):
        subj.append(k)
i2 = np.where(np.isin(skey, subj) & call_at2)[0]
y2 = (stype[i2] == "tumor").astype(int)
g2 = skey[i2]
print(f"\n=== (ii) GSE131907 同供体配对：{len(subj)} 个供体, n={len(i2)} 个 AT2 注释细胞, "
      f"label=肿瘤侧/癌旁侧（不使用 Tumor_C* 注释）===")
sp2 = donor_paired(score_raw, subj)
tab2 = []
ref2 = np.column_stack([sp2[i2], -sp2[i2]])
a0 = lodo_auc(ref2[~np.isnan(ref2[:, 0])], y2[~np.isnan(ref2[:, 0])], g2[~np.isnan(ref2[:, 0])])
tab2.append(dict(space="[ref] marker panel (donor-paired)", LODO_AUROC=a0[0], sd=a0[1], folds=a0[2]))
for tag, E in SP.items():
    ok = ~np.isnan(sp2[i2])
    auc = lodo_auc(E[i2][ok], y2[ok], g2[ok])
    rho = float(spearmanr(E[i2] @ (E[i2][y2 == 1].mean(0) - E[i2][y2 == 0].mean(0)), sp2[i2]).correlation) \
        if ok.sum() > 100 else float("nan")
    tab2.append(dict(space=tag, LODO_AUROC=auc[0], sd=auc[1], folds=auc[2],
                     spearman_vs_marker=round(rho, 3)))
results["ii_GSE131907_within_subject"] = tab2
print(pd.DataFrame(tab2).sort_values("LODO_AUROC", ascending=False).to_string(index=False))

json.dump({"panel_AT2": AT2, "panel_tumor": TU, "n_paired_subjects": len(subj),
           "i": results["i_GSE148071_cross_donor"], "ii": results["ii_GSE131907_within_subject"]},
          open(f"{RES}/marker_benchmark.json", "w"), indent=2)
pd.DataFrame(results["i_GSE148071_cross_donor"]).assign(design="i_cross_donor").to_csv(
    f"{RES}/marker_benchmark.csv", index=False)
print("\nwrote", f"{RES}/marker_benchmark.json")
