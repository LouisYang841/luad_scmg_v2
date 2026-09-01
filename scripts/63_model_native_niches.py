#!/usr/bin/env python3
"""63_model_native_niches.py — 用模型自带映射重做空转生态位，并与聚类标签版对比。

流程（全部不使用传统聚类标签作为语义来源）：
  1. 给 316k 图谱细胞一个**模型原生**身份：
       非上皮 = SCMG 全局参考软迁移的 major_cell_type（经策展归并）
       上皮   = 到参考正常上皮的距离判据（正常 / 异常）  <- 与注释和 CNV 都无关
  2. 交叉表 M[旧的30个去卷积标签 -> 原生身份]：把携带某旧标签的图谱细胞的原生身份分布
     作为该列的含义修正；spot_native = A_old @ M。
     （A_old 是现成去卷积结果，本机没有空转原始矩阵，所以只能做到这一层；
      这一步是"通过模型做标签重映射"，不是重新去卷积——已在报告里写明这个边界。）
  3. 空间邻域：每张切片按像素坐标建 spot kNN(k=5+1)，邻域平均组成 = 该 spot 的生态位特征
  4. 切片内 CLR 中心化（消掉切片/平台效应）-> PCA -> k-means，k 由轮廓系数选
  5. 与旧的 6 个手写生态位做对比：切片级 Spearman、ARI、以及对 stage 的关联强度
  6. 关键消融：只把"上皮"那一半换成原生判定，其余不变，看旧生态位结论还剩多少
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, fisher_exact
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score as ari, normalized_mutual_info_score as nmi

warnings.filterwarnings("ignore")
LUAD_INV = "/home/ubuntu/luad_invasion"
RES = L.RES
OUT = f"{RES}/niches"
os.makedirs(OUT, exist_ok=True)
K_NICHE = None  # 由轮廓系数选

# ---------------- 1. 图谱细胞的模型原生身份 ----------------
nat = pd.read_csv(f"{RES}/model_native/cell_native_annotations.csv")
epi = pd.read_csv(f"{RES}/model_native/epithelium_calls.csv")
epi["barcode"] = epi.barcode.astype(str)
abn = dict(zip(epi.barcode, epi.abn95.values))
dist = dict(zip(epi.barcode, epi.d_to_normal_epithelium.values))
nat["barcode"] = nat.barcode.astype(str)
nat["is_epi"] = nat.compartment_clean.isin(["malignant_epithelium", "normal_alveolar_epithelial"]) | \
    (nat.broad_cell_type == "Malignant")
nat["abn95"] = [abn.get(b, np.nan) for b in nat.barcode]

MAJ2G = {"T cell": "T cell", "Natural killer": "NK/ILC", "B cell": "B cell",
         "Macrophage": "Macrophage", "Dendritic cell": "Dendritic cell",
         "Endothelial": "Endothelial", "Pericyte": "Pericyte", "Smooth muscle": "Smooth muscle",
         "Stromal cell": "Fibroblast/Stromal", "Granulocyte": "Granulocyte/Mast",
         "Other blood cell": "Plasma cell", "Erythroid": "Erythroid",
         "Respiratory system": "Epithelium", "Other epithelium": "Epithelium",
         "Epithelial progenitor": "Epithelium", "Glia": "Neuron/Glia", "Neuron": "Neuron/Glia",
         "Neural progenitor": "Neuron/Glia"}


def group_of(r):
    g = MAJ2G.get(r.native_major, "Other/Embryonic")
    if g == "Epithelium":
        a = r.abn95
        if pd.isna(a):                                   # 参考距离没算到的细胞：按原注释兜底
            g = "Epithelium-abnormal" if r.compartment_clean == "malignant_epithelium" else "Epithelium-normal"
        else:
            g = "Epithelium-abnormal" if a == 1 else "Epithelium-normal"
    if g == "Plasma cell" and r.broad_cell_type == "Plasma cell":
        g = "Plasma cell"
    return g


nat["native_group"] = nat.apply(group_of, axis=1)
print("模型原生身份分布:")
print(nat.native_group.value_counts().to_string())

# ---------------- 2. 交叉表 M ----------------
OLD = ['Tumor_Tumor C3', 'Tumor_Tumor C5', 'Tumor_Tumor C0', 'Tumor_Tumor C6', 'Tumor_Tumor C9',
       'Tumor_Tumor C1', 'Tumor_Tumor C13', 'Tumor_Tumor C7', 'Tumor_Tumor C8', 'Tumor_Tumor C4',
       'Tumor_Tumor C10', 'Tumor_Tumor C12', 'Tumor_Tumor C11', 'Tumor_Tumor C2', 'CD4+ Naive/Mem T',
       'CD8+ Exhausted T', 'CD8+ Cytotoxic T', 'NK Cells', 'SPP1+ TAM', 'C1QC+ TAM',
       'Pro-inflammatory M1', 'Dendritic Cells', 'Naive B Cells', 'Memory B Cells', 'IgG+ Plasma',
       'IgA+ Plasma', 'myCAF', 'iCAF', 'Capillary EC', 'Tumor EC (Tip-like)']
BRIDGE = {                                            # spot 列名 -> 图谱 fine 标签集合
    "CD4+ Naive/Mem T": ["Effector CD4+ T cell", "Exhausted Treg", "Inflammatory Treg"],
    "CD8+ Cytotoxic T": ["Effector CD8+ T cell", "DNT"],
    "CD8+ Exhausted T": ["Exhausted CD8+ T cell"],
    "Memory B Cells": ["Memory B cell", "Proliferative B cell", "Transitional B cell"],
    "Naive B Cells": ["Naive B cell"],
    "Normal Alveolar Epithelial": ["Normal Alveolar Epithelial"],
}
groups = sorted(nat.native_group.unique())
M = np.zeros((len(OLD), len(groups)))
cover = {}
for i, lab in enumerate(OLD):
    src = BRIDGE.get(lab, [lab])
    sel = nat[nat.assigned_fine_label.isin(src)]
    cover[lab] = len(sel)
    if len(sel) == 0:                                  # 图谱里没有 -> 恒等透传并标记
        j = groups.index("Other/Embryonic")
        M[i, j] = 1.0
    else:
        vc = sel.native_group.value_counts()
        for g, c in vc.items():
            M[i, groups.index(g)] = c
        M[i] /= M[i].sum()
print("\n交叉表覆盖度（图谱里能找到的细胞数，0 = 透传）:")
print(json.dumps({k: int(v) for k, v in cover.items()}, indent=0, ensure_ascii=False))

# ---------------- 3. spot 组成（从全部切片文件装配，master 表只有 7 张） ----------------
SDE = f"{LUAD_INV}/results/spatial_deconv_abundance"
gt = pd.read_csv(f"{LUAD_INV}/data/spatial_15slides_ground_truth_manifest.csv")
gt["slide_id"] = gt["slide_id"].astype(str)
key = gt.set_index("slide_id")[["dataset", "patient_id", "histology", "platform", "stage"]]
ALT = {"P4_AAH1": "P4_AAH", "P4_AAH2": "P4_AAH-1"}      # 文件名 -> manifest slide_id
OLDNICHESLIDES = pd.read_csv(f"{LUAD_INV}/results/spatial_niches/"
                             "Visium_14Slides_Spatial_Niche_Dynamics.csv")["Slide_ID"].tolist()
frames = []
for f in sorted(os.listdir(SDE)):
    if not f.endswith(".csv") or "master" in f:
        continue
    sl = f.replace("_seurat_transfer_abundance.csv", "")
    d = pd.read_csv(f"{SDE}/{f}")
    d.columns = [c.strip().strip('"') for c in d.columns]
    miss = [c for c in OLD if c not in d.columns]
    for c in miss:
        d[c] = 0.0
    d = d.rename(columns={"ident": "spot_barcode"})
    if "spot_barcode" not in d.columns:
        d["spot_barcode"] = d.iloc[:, 0].astype(str)
    d["slide"] = sl
    d["stage_file"] = d.get("stage", pd.Series(["NA"] * len(d))).values
    frames.append(d)
m = pd.concat(frames, ignore_index=True)
m = m[m.slide.isin(OLDNICHESLIDES)].copy().reset_index(drop=True)   # 与旧生态位表同一批 15 张切片
sid = m.slide.map(lambda x: ALT.get(x, x))
for c in ["dataset", "patient_id", "histology", "platform"]:
    m[c] = sid.map(key[c]).values
m["gt_stage"] = sid.map(key["stage"]).values        # ground-truth manifest 的组织学分期
print(f"\n装配后 spots {m.shape[0]}  slides {m.slide.nunique()}  "
      f"spatial datasets {m.dataset.nunique()}  patients {m.patient_id.nunique()}")
print("\n⚠ 旧 niche 表的 stage 与 ground-truth manifest 的冲突:")
cf = pd.DataFrame({"gt_stage": key["stage"]})
chk = m.groupby("slide").agg(gt_stage=("gt_stage", "first"), file_stage=("stage_file", "first"),
                             dataset=("dataset", "first"), patient=("patient_id", "first"))
dfr = chk[chk.gt_stage.astype(str) != chk.file_stage.astype(str)]
print("  冲突切片数 = %d / %d" % (len(dfr), len(chk)))
if len(dfr):
    print(dfr.to_string())
    print("  -> 下面一律用 ground-truth 的 gt_stage")
print("\n每 stage 的切片/供体数（真实功效）:")
print(m.groupby("gt_stage").agg(slides=("slide", "nunique"), patients=("patient_id", "nunique"),
                                spots=("slide", "size")).to_string())

slide = m["slide"].astype(str).values
sp_ds = m["dataset"].values
sp_pat = m["patient_id"].values
stage = m["gt_stage"].values
A = m[OLD].values.astype(np.float32)
A = A / (A.sum(1, keepdims=True) + 1e-9)
S = A @ M
S = S / (S.sum(1, keepdims=True) + 1e-9)

# ---------------- 4. 空间邻域生态位特征 ----------------
def neighbor_profile(mat, slide, m):
    """每张切片按像素坐标做 spot kNN(k=6)，返回邻域平均组成 + 无坐标切片清单。"""
    out = np.zeros_like(mat)
    bad = []
    for sl in pd.unique(slide):
        i = np.where(slide == sl)[0]
        xy = m.loc[i, ["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(float)
        ok = np.isfinite(xy).all(1)
        if ok.sum() < 20:
            bad.append((sl, len(i), int(ok.sum())))
            out[i] = mat[i]                       # 退化：用自身组成
            continue
        i, xy = i[ok], xy[ok]
        _, nb = cKDTree(xy).query(xy, k=min(6, len(xy)))
        out[i] = mat[i][np.atleast_2d(nb)].mean(1)
        rest = np.where(slide == sl)[0]
        miss = np.setdiff1d(rest, i)
        if len(miss):
            out[miss] = mat[miss]
    return out, bad


prof, bad = neighbor_profile(S, slide, m)
if bad:
    print("\n⚠ 无有效空间坐标、退化为单 spot 组成的切片:", bad)
# 切片内 CLR 中心化（消掉切片/平台组成差异）
def clr_center(prof, slide):
    Xc = np.zeros_like(prof)
    for sl in pd.unique(slide):
        i = np.where(slide == sl)[0]
        lc = np.log(prof[i] + 1e-6)
        Xc[i] = lc - lc.mean(1, keepdims=True)
    return Xc


def niches_within_slide(prof, slide, k=6, seed=0):
    """切片内聚类 -> 每片 k 个簇；再用各簇的原生组成质心做跨切片匹配，得到全局生态位原型。"""
    Xc = clr_center(prof, slide)
    local = np.full(len(slide), -1, dtype=int)
    cent = []
    for sl in pd.unique(slide):
        i = np.where(slide == sl)[0]
        if len(i) < 50:
            local[i] = 0; cent.append(np.zeros((1, prof.shape[1]))); continue
        km = KMeans(k, n_init=10, random_state=seed).fit(Xc[i])
        local[i] = km.labels_
        cent.append(prof[i][km.labels_ == -1] if False else
                    np.stack([prof[i][km.labels_ == c].mean(0) for c in range(k)]))
    cent = np.concatenate(cent)                                # (nslides*k, groups)
    C = cent / (cent.sum(1, keepdims=True) + 1e-9)
    Cd = PCA(n_components=min(8, C.shape[1], C.shape[0] - 1), random_state=0,
             svd_solver="full").fit_transform(C)
    Cd = Cd / (Cd.std(0) + 1e-9)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    Zl = linkage(pdist(Cd), method="ward")
    arch = fcluster(Zl, t=n_archetypes, criterion="maxclust") - 1
    glob = np.zeros(len(slide), dtype=int)
    r = 0
    for sl in pd.unique(slide):
        for c in range(k):
            i = np.where((slide == sl) & (local == c))[0]
            if len(i):
                glob[i] = arch[r]
            r += 1
    return glob, arch


n_archetypes = 8
lab, arch = niches_within_slide(prof, slide, k=6)
K_NICHE = len(np.unique(lab))
print(f"\n切片内 k=6 + 跨切片原型匹配 -> 全局生态位原型 {K_NICHE} 个")

# ---------------- 5b. 消融：同样的算法，只把"上皮"换回原聚类标签 ----------------
M_old = np.zeros_like(M)
iepi = [g for g in groups if g.startswith("Epithelium")]
for i, lb in enumerate(OLD):
    if lb.startswith("Tumor_Tumor"):
        M_old[i, groups.index("Epithelium-abnormal")] = 1.0
    elif lb == "Normal Alveolar Epithelial":
        M_old[i, groups.index("Epithelium-normal")] = 1.0
    else:
        M_old[i] = M[i]
S2 = A @ M_old
S2 /= (S2.sum(1, keepdims=True) + 1e-9)
prof2, _ = neighbor_profile(S2, slide, m)
lab2, _ = niches_within_slide(prof2, slide, k=6)
print(f"消融（同一算法，只换回原上皮标签）: ARI={ari(lab, lab2):.3f}  NMI={nmi(lab, lab2):.3f}")

# ---------------- 6. 与旧生态位对比 + stage 关联 ----------------
newf = pd.DataFrame(S, columns=groups)
newf["slide"] = slide; newf["niche"] = lab; newf["stage"] = stage
newf["sp_dataset"] = sp_ds; newf["sp_patient"] = sp_pat
frac = newf.groupby(["slide", "niche"]).size().unstack(fill_value=0)
frac = frac.div(frac.sum(1), axis=0).rename(columns=lambda c: f"archetype{int(c)}")
comp = newf.groupby("slide")[groups].mean()
frac.to_csv(f"{OUT}/slide_x_niche_native.csv")
comp.to_csv(f"{OUT}/slide_x_nativecomposition.csv")

oldn = pd.read_csv(f"{LUAD_INV}/results/spatial_niches/Visium_14Slides_Spatial_Niche_Dynamics.csv")
oldn = oldn.set_index("Slide_ID")
common = [s for s in oldn.index if s in frac.index]
OLDNICHES = [c for c in oldn.columns if c not in ("Stage", "N_Spots")]
tgt = pd.concat([frac, comp], axis=1)
allcols = list(tgt.columns)
cor = pd.DataFrame(index=OLDNICHES, columns=allcols, dtype=float)
for oc in OLDNICHES:
    for gc in allcols:
        cor.loc[oc, gc] = spearmanr(oldn.loc[common, oc].values, tgt.loc[common, gc].values).correlation
cor = cor.astype(float)
cor.to_csv(f"{OUT}/old_vs_new_spearman.csv")
# 每个原型的定义（原生组成的均值，供解读）
arch_def = newf.groupby("niche")[groups].mean()
arch_def["n_spots"] = newf.groupby("niche").size()
arch_def.to_csv(f"{OUT}/archetype_definitions.csv")
print("\n=== 新生态位原型的定义（按模型原生身份的平均占比）===")
print(arch_def.round(3).sort_values("n_spots", ascending=False).to_string())
print("\n=== 旧 6 手写生态位 vs 新生态位/新原生组成（切片级 Spearman, n=%d）===" % len(common))
for oc in OLDNICHES:
    r = cor.loc[oc].dropna().sort_values(key=abs, ascending=False)
    top = ", ".join(f"{g}:{r[g]:+.2f}" for g in r.index[:4])
    print(f"  {oc:32s} -> {top}")
print("\n每个旧手写 score 与新变量的最大|r|:")
print(cor.abs().max(1).round(2).to_string())
print("\n每个新原型/原生组成与旧 6 个的最大|r|:")
print(cor.abs().max(0).round(2).to_string())

# ---- 供体内配对：同一供体的癌前区域 vs 侵袭区域（GSE307534 的 P1-P3）----
PRE = {"Normal", "AAH", "AIS", "MIA"}
comp_cell = newf.groupby(["sp_patient", "stage"])[groups].mean()
arch = newf.groupby(["sp_patient", "stage", "niche"]).size().unstack(fill_value=0)
arch = arch.div(arch.sum(1), axis=0)
pairs = []
for pt in pd.unique(newf.sp_patient):
    stg = sorted(set(newf.loc[newf.sp_patient == pt, "stage"]))
    if any(x in PRE for x in stg) and "IAC" in stg:
        pairs.append(pt)
print("\n=== 同一供体『癌前/正常区域 vs IAC 区域』配对（真实可识别设计）===")
print("可用供体:", pairs, " n=%d  ⚠ n 极小，只能当方向性参考" % len(pairs))
if pairs:
    rows = []
    for g in groups + [f"niche{c}" for c in sorted(newf.niche.unique())]:
        src = comp_cell if g in groups else arch
        d = []
        for pt in pairs:
            try:
                a = src.loc[(pt, [x for x in src.loc[pt].index if x in PRE][0]), g]
                b = src.loc[(pt, "IAC"), g]
                d.append(float(b) - float(a))
            except Exception:
                pass
        if d:
            rows.append(dict(feature=g, mean_delta_IAC_minus_pre=round(float(np.mean(d)), 4),
                             n_pairs=len(d), all_same_sign=bool(np.all(np.array(d) > 0) or np.all(np.array(d) < 0))))
    pr = pd.DataFrame(rows).sort_values("mean_delta_IAC_minus_pre", key=abs, ascending=False)
    print(pr.to_string(index=False))
    pr.to_csv(f"{OUT}/within_patient_pre_vs_IAC.csv", index=False)

st = newf.groupby(["stage", "niche"]).size().unstack(fill_value=0)
st = st.div(st.sum(1), axis=0)
print("\n=== 新生态位 × stage（spot 级占比）===")
print(st.round(3).to_string())
print("\n=== 每 stage 的真实功效（spot / 切片 / 供体）===")
print(newf.groupby("stage").agg(spots=("niche", "size"), slides=("slide", "nunique"),
                                patients=("sp_patient", "nunique")).to_string())
print("\n=== 供体级生态位占比（这才是推断单元；每供体切片数<=2）===")
dp = newf.groupby(["sp_patient", "niche"]).size().unstack(fill_value=0)
dp = dp.div(dp.sum(1), axis=0)
dp["stage"] = newf.groupby("sp_patient")["stage"].first()
dp["n_spots"] = newf.groupby("sp_patient").size()
print(dp.round(3).sort_values("stage").to_string())
dp.to_csv(f"{OUT}/patient_x_niche_native.csv")

pd.DataFrame({"spot_barcode": m.spot_barcode, "slide": slide, "stage": stage,
              "sp_dataset": sp_ds, "sp_patient": sp_pat,
              "niche_native": lab, "niche_ablation_oldepi": lab2}).to_csv(f"{OUT}/spot_niches.csv", index=False)
json.dump({"K_niche": int(K_NICHE), "silhouette": float("nan"), "groups": groups,
           "ari_native_vs_oldepi_ablation": float(ari(lab, lab2)),
           "n_spots": int(len(S)), "n_slides": int(len(np.unique(slide)))},
          open(f"{OUT}/niche_summary.json", "w"), indent=2)
print("\nwrote", OUT)
