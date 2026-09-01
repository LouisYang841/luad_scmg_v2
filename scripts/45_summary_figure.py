#!/usr/bin/env python3
"""45_summary_figure.py — 一张图说清：适应 vs 流形保持的权衡，以及"表观提升"从哪来。"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

RES = L.RES
plt.rcParams.update({"figure.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
fig.suptitle("LUAD x SCMG fine-tune rework: real gains vs label leakage", fontsize=12, fontweight="bold")

# ---- (a) epoch 扫描：held-out 掩码基因恢复 vs 与全局流形的偏离 ----
s = pd.read_csv(f"{RES}/sweep_metrics.csv")
h = pd.DataFrame(json.load(open(f"{RES}/finetune_unsup/history.json")))
ax = axes[0, 0]
x = np.arange(len(s))
ax.plot(x, s["mask_recovery_corr_luadTok"], "o-", color="#0b6e4f", label="held-out masked-gene recovery (r)")
ax.set_ylabel("masked-gene recovery (held-out donors)", color="#0b6e4f")
ax2 = ax.twinx()
ax2.plot(x[1:], h["val_c"], "s--", color="#c1121f", ms=4, label="val_c on frozen zero-shot graph (drift)")
ax2.set_ylabel("val_c on frozen zero-shot graph", color="#c1121f")
ax.axvline(1, color="grey", lw=1, ls=":")
ax.annotate("chosen: ep1\nmore training = donor overfit", (1, 0.6569), xytext=(4.2, 0.648),
            arrowprops=dict(arrowstyle="->", color="grey"), fontsize=8)
ax.set_xlabel("epoch")
ax.set_title("(a) unsupervised adaptation: gain only in epoch 1", loc="left")
ax.set_xticks(x[::2]); ax.set_xticklabels(s["model"][::2], rotation=45, ha="right", fontsize=7)

# ---- (b) 诚实 vs 循环的评估 ----
A = json.load(open(f"{RES}/audit_prev_work.json"))
t = pd.DataFrame(A["1_stage_probe_leakage"])
ax = axes[0, 1]
w = 0.36; xr = np.arange(len(t))
ax.bar(xr - w / 2, t["LOPO_unseen_donors"], w, color="#1d3557", label="unseen donors (honest)")
ax.bar(xr + w / 2, t["random_split_sample_leak"], w, color="#e63946", alpha=.8, label="random split (donor leakage)")
ax.axhline(A["1_chance"], color="grey", ls="--", lw=1)
ax.text(2.4, A["1_chance"] + .02, "chance", fontsize=7, color="grey")
ax.set_xticks(xr); ax.set_xticklabels([m.replace("(leaky)", "\n(leaky)") for m in t["model"]], fontsize=8)
ax.set_ylabel("balanced accuracy (stage, 6 classes)")
ax.set_title("(b) where the old stage-gain came from", loc="left")
ax.legend(fontsize=7, frameon=False)
for i, (a_, b_) in enumerate(zip(t["LOPO_unseen_donors"], t["random_split_sample_leak"])):
    ax.text(i - w / 2, a_ + .01, f"{a_:.3f}", ha="center", fontsize=7)
    ax.text(i + w / 2, b_ + .01, f"{b_:.3f}", ha="center", fontsize=7)

# ---- (c) 扩散"轨迹"是连线假象 ----
g = np.load("/tmp/diffusion_transition_100.npy")
from sklearn.decomposition import PCA
emb = np.load("/tmp/emb_stage_mal.npy")
p = PCA(2, random_state=0).fit(emb[np.random.default_rng(0).choice(len(emb), 6000)])
r = p.transform(emb); q = p.transform(g)
ax = axes[1, 0]
ax.scatter(r[:, 0], r[:, 1], s=1, alpha=.06, color="#adb5bd", lw=0)
ax.plot(q[:, 0], q[:, 1], "-", color="#ffb703", lw=1.2, alpha=.9, label="old 'Normal->IAC trajectory'")
ax.scatter(q[:, 0], q[:, 1], s=14, color="#fb8500", lw=0, zorder=5)
ax.set_title("(c) adjacent/all-pairs dist ratio 0.97 -> iid draws, not a path", loc="left", fontsize=9.5)
ax.legend(fontsize=7, frameon=False, loc="best")
ax.set_xticks([]); ax.set_yticks([])

# ---- (d) 四个空间的正面比较 ----
c = pd.read_csv(f"{RES}/comparison_models_final.csv")
ax = axes[1, 1]
met = ["mask_recovery_corr_luadTok", "recon_corr", "LOPO_malignant_vs_AT2__GSE148071", "LOPO_celltype6"]
lbl = ["held-out\nmask recovery", "recon\ncorr", "held-out donor\nmalignancy bAcc", "held-out donor\ncell-type bAcc"]
cols = {"ZS": "#457b9d", "FT-luad(leaky)": "#e63946", "FT-stage(leaky)": "#f4a261",
        "FT-unsup(ep1)": "#2a9d8f", "FT-unsup(ep15)": "#9d4edd"}
xr = np.arange(len(met)); bw = 0.2
for i, (_, row) in enumerate(c.iterrows()):
    ax.bar(xr + (i - 2) * bw, [row[m] for m in met], bw, color=cols.get(row["model"], "#999"),
           label=row["model"], edgecolor="white", lw=.4)
ax.set_xticks(xr); ax.set_xticklabels(lbl, fontsize=8)
ax.set_title("(d) only ep1 wins all three held-out metrics; ep15 degrades", loc="left", fontsize=9.5)
ax.legend(fontsize=6.5, frameon=False, ncol=2)
ax.set_ylim(0, 1.02)

plt.tight_layout(rect=[0, 0, 1, .96])
plt.savefig(f"{RES}/summary_rework.png", bbox_inches="tight")
print("wrote", f"{RES}/summary_rework.png")
