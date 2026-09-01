#!/usr/bin/env python3
"""65_niche_figure.py — 生态位对比图 + 把生态位与上皮异常度结果并入 REPORT.md（第 8、9 节）。"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

RES, OUT = L.RES, f"{L.RES}/niches"
ad = pd.read_csv(f"{OUT}/archetype_definitions.csv", index_col=0)
cor = pd.read_csv(f"{OUT}/old_vs_new_spearman.csv", index_col=0)
summ = json.load(open(f"{OUT}/niche_summary.json"))
EN = json.load(open(f"{RES}/model_native/epithelium_novelty.json"))
ec = pd.read_csv(f"{RES}/model_native/epithelium_calls.csv")

groups = [g for g in ad.columns if g != "n_spots"]
old_rows = [r for r in cor.index if not r.startswith("N_")]
A = ad[groups].loc[ad.sort_values("n_spots", ascending=False).index]
comp_cols = [c for c in cor.columns if not c.startswith("archetype")]
arch_cols = [c for c in cor.columns if c.startswith("archetype")]
ctab = cor.loc[old_rows, comp_cols].astype(float).round(2).to_string()
atab = cor.loc[old_rows, arch_cols].astype(float).round(2).to_string()

# --------------------------------------------------------------- figure --
plt.rcParams.update({"figure.dpi": 150, "font.size": 8})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
fig.suptitle("Spatial niches from SCMG's own global mapping (no clustering labels) vs the label-based version",
             fontweight="bold")
ax = axes[0]
im = ax.imshow(A.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=90, fontsize=7)
ax.set_yticks(range(len(A)))
ax.set_yticklabels([f"{i} (n={int(ad.loc[i,'n_spots'])//1000}k)" for i in A.index], fontsize=7)
ax.set_title("(a) model-native niche archetypes", loc="left")
plt.colorbar(im, ax=ax, fraction=.046)

ax = axes[1]
C = cor.loc[old_rows, comp_cols].astype(float)
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(comp_cols))); ax.set_xticklabels(comp_cols, rotation=90, fontsize=6.5)
ax.set_yticks(range(len(C))); ax.set_yticklabels([c[:30] for c in C.index], fontsize=6.5)
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        v = C.values[i, j]
        if np.isfinite(v) and abs(v) > .7:
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > .85 else "black")
ax.set_title("(b) old hand-written niche scores vs\nmodel-native spot composition (Spearman, 15 slides)", loc="left")
plt.colorbar(im, ax=ax, fraction=.046)

ax = axes[2]
mx = C.abs().max(axis=1)
ax.barh(range(len(mx)), mx.values, color=["#c1121f" if v < .8 else "#2a9d8f" for v in mx.values])
ax.set_yticks(range(len(mx))); ax.set_yticklabels([c[:30] for c in mx.index], fontsize=6.5)
ax.invert_yaxis(); ax.set_xlabel("max |Spearman| over model-native compartments")
ax.set_title("(c) strongest anchor for each old niche score\n(red < 0.8: weakly anchored)", loc="left")
for i, v in enumerate(mx.values):
    ax.text(v + .01, i, f"{v:.2f}", va="center", fontsize=6.5)
plt.tight_layout(rect=[0, 0, 1, .92])
plt.savefig(f"{OUT}/niche_comparison.png", bbox_inches="tight")
plt.close()
print("wrote", f"{OUT}/niche_comparison.png")

# ------------------------------------------------------- report sections --
t95 = float(np.quantile(ec[(ec.dataset == "GSE131907") & (ec.site == "adjacent_normal")].d_to_normal_epithelium, .95))
g13 = ec[ec.dataset == "GSE131907"].groupby("site").agg(
    med_dist=("d_to_normal_epithelium", "median"), frac_abnormal=("abn95", "mean"),
    n=("abn95", "size"), donors=("patient_id", "nunique")).round(3)

sec8 = f"""
## 8. 空转生态位：用模型自带映射重做，并与聚类标签版对比

- 切片 {summ['n_slides']} 张 / spot {summ['n_spots']:,} 个 / **供体仅 10 个**；模型原生身份 {len(groups)} 类
- 做法：切片内 k=6 聚类 + 跨切片原型匹配（避免"聚类只是在恢复哪张切片"）
- **消融（完全同一算法，只把上皮换回原聚类标签）：ARI = {summ['ari_native_vs_oldepi_ablation']:.3f}**
  → 生态位划分几乎完全由"上皮怎么标"决定，而那一步已证明在 GSE131907 上是错的（见第 9 节）。

### 8a. 旧 6 个"手写 score 生态位" vs 模型原生成分（与聚类无关，最稳健）

```
{ctab}
```

- 四个旧生态位在原生标签下方向一致且强对应：Lepidic↔正常上皮 +0.78；myCAF 屏障↔Pericyte/平滑肌/成纤维 +0.96~0.98；
  Immune-hotspot↔T +0.90（且 NK +0.88 / B +0.84）；TAM 屏障↔巨噬 +0.90。**这四个可以保留。**
- `Invasive_Solid_Core_Niche` 对上的是 `Other/Embryonic`（rho=+0.97）。查了这个桶的真身：
  主要是参考图谱里的**表皮/基底样上皮**细胞（13,133 个）。对 LUAD 实体/鳞状化区来说这是**合理的生物学信号**，
  不是垃圾桶——模型的自带映射独立地把这些细胞判成"非肺来源的上皮样程序"，与旧 score 指的同一批 spot。
  但同一桶里还混着 8,575 个被注释成 normal AT2 的细胞，所以它的准确含义是
  **"没有正常肺对应物的上皮"**，而不是"鳞状"。
- `Exhausted_TME_Niche` 与 `Epithelium-normal` 是 **-0.89**：它基本就是"正常上皮少"的同义词，
  本身不携带任何 T 细胞耗竭信息（与 T cell 只有 +0.74，且与 B/Plasma 更强）→ **这个名字是过度解释**。
- **重要边界：`argmax` 原生类型对肿瘤细胞不可信**（参考里没有肿瘤数据集，会被扒到前列腺/胸腺/牙等离谱组织）；
  可信的是距离量（第 9 节）。

### 8b. 生态位原型级的对应（离散簇）

```
{atab}
```

离散原型与旧 score 的对应明显弱于成分级（最大 0.75，Lepidic 与 Immune-hotspot 只有 0.28~0.35）
→ 旧的"生态位"本质是组成的线性组合，不是空间上独立存在的尖锐类别。

### 8c. 必须先修的两个数据问题

1. `all_slides_spatial_deconv_master.csv` **只并了 GSE307534 的 7 张切片**；旧生态位表用的是 15 张，
   其余 8 张从未进入 master（本轮已从单张表重建为 15 张 / 118,565 spot / 10 供体）。
2. **15 张里有 5 张的 stage 与 ground-truth manifest 矛盾**，全在 GSE189487：
   TD1、TD2 标 AIS 实为 IAC；TD5 标 MIA 实为 AIS；TD6 标 IAC 实为 MIA；TD8 标 IAC 实为 AIS。
   → 分期-生态位曲线的横轴本身就是错的，这五条切片的归属要改。

### 8d. 方法学问题：旧的 6 个"生态位"根本不是聚类出来的

`12_spatial_cellular_neighborhoods_and_niches.py` 里这 6 个变量是对 Seurat 转移概率的**手写线性组合**
（`lepidic_score` / `solid_score` / `mycaf_score` / `tex_score`），不是无监督的生态位发现。
所以"旧 score 与原生成分高度相关"是同一个东西的两种写法，**不能当作独立验证**。

### 8e. 最强设计（供体内配对）与其结果：阴性的

真正可配对的是 GSE307534 同一供体多区域：Patient_1(AAH vs IAC)、Patient_2(AAH vs IAC)、
Patient_3(AIS vs IAC) → **n=3 供体**。在这 3 个配对上，癌前区域 vs IAC 区域的所有原生成分与生态位原型占比
**|Δ| ≤ 0.018 且符号不一致** → **没有哪个生态位在供体内从癌前到侵袭发生可重复的改变**。
同时，同一批供体的**上皮异常度**却是 9/9 一致、p=0.004（第 9 节）。
→ **变的是上皮细胞本身，不是它们的生态位。**这句话是这一轮生态位分析最重要的结论。

![niche](niches/niche_comparison.png)
"""

sec9 = f"""
## 9. 模型原生的"上皮异常度"——不用注释、不用 CNV 的第三判据

裸的 novelty 不能用（被参考类别密度支配：浆细胞 5.3、myCAF 2.8 的"新奇度"比多数肿瘤 clone 还高）。
改成：只在参考图谱的**正常呼吸/上皮**细胞里找最近邻，并用参考自身 1NN 距离做密度校正；
阈值取癌旁正常上皮的 q95 = {t95:.2f}。

```
{g13.to_string()}
```

**同供体配对检验（GSE131907，{EN['n_paired_subjects']} 个供体；这些上皮全部被注释成 normal AT2）**

```
肿瘤侧更远的供体比例 = {EN['frac_tumor_farther']:.2f} | 配对差值均值 = {EN['paired_delta_mean']:.3f}
t 检验 p = {EN['paired_t_p']:.4g} | Wilcoxon p = {EN['paired_wilcoxon_p']:.4g}
供体内 AUC = {EN['auc_within_subject_mean']:.3f}（混合供体 {EN['auc_pooled']:.3f}）
```

### 结论

1. **独立复现了你朋友的 CNV 结论**：GSE131907 肿瘤样本里的上皮细胞，在不看任何注释、不看 CNV 的情况下，
   9/9 个供体都比同一供体的癌旁上皮明显更靠离正常肺上皮流形（p=0.004）。
   所以"GSE131907 有 0 个恶性上皮"是**注释错误**，不是数据里没有肿瘤。
2. 分化梯度（同一 dataset 内）：癌旁 5% → 原发肿瘤 19% → 淋巴结转移 46% → 胸水转移 56% 被判为异常上皮。
3. **但跨 dataset 不可比**：同样是 annotated malignant，GSE189357 中位距离 0.96，GSE148071 是 2.69。
   平台/处理差异会盖过生物学差异 → 这个判据只能用于**同一 dataset 内、同供体配对**。
4. 因此正确的分工不变：**区室用 SCMG 全局映射；恶性用 CNV；这个距离量做第三方交叉验证。**
   等他的 CNV 表（barcode + score + call + 方法 + 参考细胞定义）一到，直接算 novelty vs CNV call 的 AUROC。
"""

txt = open(f"{RES}/REPORT.md").read()
for tag in ["\n## 8.", "\n## 9."]:
    c = txt.find(tag)
    if c > 0:
        txt = txt[:c]
open(f"{RES}/REPORT.md", "w").write(txt.rstrip() + "\n" + sec8 + sec9)
print("appended sections 8 & 9 to REPORT.md")
