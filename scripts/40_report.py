#!/usr/bin/env python3
"""40_report.py — 汇总所有产物为 results/REPORT.md（含可识别性表）。"""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

RES = L.RES
order = L.load_order()
ann = pd.read_csv(f"{RES}/cell_annotations_clean.csv", index_col=0)
md = []
def w(*a): md.append(" ".join(str(x) for x in a))

w("# LUAD × SCMG 微调返工报告\n")
w("生成时间:", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"), "\n")

# ---------- 可识别性 ----------
w("## 1. 可识别性表：哪些对比在当前数据里根本立不住\n")
st = order.groupby("stage").agg(n_cells=("patient_id", "size"), n_donors=("patient_id", "nunique"),
                                n_datasets=("dataset", "nunique"))
st["datasets"] = order.groupby("stage")["dataset"].apply(lambda x: "|".join(sorted(set(x))))
st["tissue_sites"] = order.groupby("stage")["tissue_site"].apply(lambda x: "|".join(sorted(set(x))))
w("### 1a. stage 与 dataset / 组织部位 / 供体 完全混淆\n")
w("```")
w(st.to_string())
w("```\n")
w("> AAH 只有 **1** 个供体、MIA **2** 个、AIS **3** 个；Normal 与 LNM 只来自 GSE131907，"
  "AAH/AIS/MIA 只来自 GSE189357。所以『细胞级 stage 分离度』衡量的是 dataset/供体身份，不是进展。\n")

comp = pd.crosstab(ann["broad_cell_type"], ann["compartment_clean"])
w("### 1b. 上一轮的 `Malignant` 区室 = Pure_Tumor 子图谱成员身份，不是恶性细胞\n")
w("```"); w(comp.to_string()); w("```\n")
cds = pd.crosstab(ann.loc[ann["broad_cell_type"] == "Malignant", "compartment_clean"],
                  ann.loc[ann["broad_cell_type"] == "Malignant", "dataset"])
w("```"); w(cds.to_string()); w("```\n")
w("> GSE131907 贡献的 23,350 个『肿瘤细胞』里，真正的恶性上皮 = **0**；"
  "GSE189357 的 8,876 个则 100% 是恶性上皮。也就是说 `broad_cell_type` 的含义随 dataset 漂移。\n")

w("### 1c. 唯一可做『供体内配对』的恶性对比\n")
both = [d for d in order["patient_id"].unique()
        if ((order["patient_id"] == d) & (order["malignant_call"] == "malignant")).sum() >= 30
        and ((order["patient_id"] == d) & (order["malignant_call"] == "normal_at2")).sum() >= 30]
w(f"- 同时含 malignant(>=30) 与 normal AT2(>=30) 的供体：**{len(both)}** 个")
w(f"- 其中 GSE148071 {sum('_P' in b for b in both)} 个，其他 {sum('_P' not in b for b in both)} 个")
w(f"- malignant 细胞 {int((order['malignant_call']=='malignant').sum())} / "
  f"normal AT2 {int((order['malignant_call']=='normal_at2').sum())} / 全部 {len(order)}\n")

# ---------- 上一轮审计 ----------
if os.path.exists(f"{RES}/audit_prev_work.json"):
    A = json.load(open(f"{RES}/audit_prev_work.json"))
    w("## 2. 对上一轮工作的可复现审计 (`scripts/25_audit_prev_work.py`)\n")
    w("### 2a. 循环评估：换成未见供体后，『stage 微调』的增益归零\n")
    w("```")
    w(pd.DataFrame(A["1_stage_probe_leakage"]).to_string(index=False))
    w("```\n")
    w(f"chance = {A['1_chance']}；留出供体 = {A['1_held_out_donors']} 个。"
      "`random_split_sample_leak` 是上一轮评估的等价物（同供体同时出现在 train/test）。\n")
    w("### 2b. 扩散『Normal→IAC 轨迹』是连线的假象\n")
    w("```")
    for k, v in A["3_diffusion_trajectory"].items():
        w(f"{k}: {v}")
    w("```\n")
    w("### 2c. encoder 被改动后的生成-解释链路\n")
    w("```")
    for k, v in A["4_recon_corr"].items():
        w(f"{k}: {v}")
    w("```")
    w(f"\n`{A['4d_base_is_not_a_lung_cancer_model']}`\n")

# ---------- 本轮训练 ----------
H = f"{RES}/finetune_unsup/history.json"
if os.path.exists(H):
    h = pd.DataFrame(json.load(open(H)))
    w("## 3. 本轮无监督微调的训练/验证曲线\n")
    w("```")
    w(h[["epoch", "train_c", "train_rec", "val_c", "val_rec", "val_recon_corr",
         "eff_rank_val", "mean_norm_val"]].round(3).to_string(index=False))
    w("```")
    w("\n> `val_c` 是在**固定的 zero-shot kNN 图**上算的对比损失，因此它衡量的是"
      "与全局流形的偏离程度（越大越偏离），不是拟合质量；不能用来选 checkpoint。"
      "选型改用下面的 masked-gene recovery（无标签泛化指标）。\n")

if os.path.exists(f"{RES}/sweep_metrics.csv"):
    s = pd.read_csv(f"{RES}/sweep_metrics.csv")
    w("## 4. epoch 扫描：适应度 vs 流形保持度的权衡\n")
    cols = [c for c in ["model", "recon_corr", "mask_recovery_corr_luadTok",
                        "mask_recovery_rmse_luadTok", "mask_recovery_corr_lungTok",
                        "recon_corr_lungtok", "mean_norm"] if c in s.columns]
    w("```"); w(s[cols].to_string(index=False)); w("```")
    if "mask_recovery_corr_luadTok" in s:
        b = s.sort_values("mask_recovery_corr_luadTok").iloc[-1]
        w(f"\n>>> 按 masked-gene recovery 最佳的 epoch: **{b['model']}** "
          f"(corr={b['mask_recovery_corr_luadTok']:.4f})\n")

files = sorted(glob.glob(f"{RES}/comparison_*.csv"))
if files:
    c = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(subset=["model"])
    w("## 5. \u591a\u4e2a\u7a7a\u95f4\u7684\u6b63\u9762\u6bd4\u8f83\uff08\u5168\u90e8\u5728**\u7559\u51fa\u4f9b\u4f53**\u4e0a\u7b97\uff09\\n")
    keep = [x for x in ["model", "mask_recovery_corr_luadTok", "recon_corr",
                        "LOPO_malignant_vs_AT2__GSE148071", "LOPO_celltype6",
                        "LOPO_stage6_CONFOUNDED", "sil_stage6_leaky",
                        "CKA_vs_ZS", "knn20_overlap_vs_ZS", "eff_rank", "mean_norm"] if x in c.columns]
    w("```"); w(c[keep].to_string(index=False)); w("```")
    w("\n> `sil_stage6_leaky` \u5c31\u662f\u4e0a\u4e00\u8f6e\u5f53\u300c\u6539\u8fdb\u8bc1\u636e\u300d\u7528\u7684\u90a3\u7c7b\u6307\u6807\uff08\u6807\u7b7e\u65e2\u53c2\u4e0e\u8bad\u7ec3\u53c8\u53c2\u4e0e\u6253\u5206\uff09\uff0c\u4fdd\u7559\u5728\u6b64\u53ea\u4e3a\u5c55\u793a\u5b83\u4e0e\u5176\u4ed6\u5217\u4e0d\u540c\u6b65\uff1b\u5176\u4f59\u5217\u90fd\u662f\u4f9b\u4f53\u7ea7\u7559\u51fa\u3002")
    w("> \u53e6\u6ce8\u610f CKA \u5bf9\u5c40\u90e8\u7ed3\u6784\u6539\u53d8\u4e0d\u654f\u611f\uff08leaky \u6a21\u578b CKA \u4ecd\u9ad8\u8fbe 0.96\uff09\uff0ckNN20 \u91cd\u53e0\u624d\u662f\u66f4\u80fd\u53cd\u6620\u6d41\u5f62\u91cd\u6392\u7684\u91cf\u3002\\n")

if os.path.exists(f"{RES}/marker_benchmark.json"):
    M = json.load(open(f"{RES}/marker_benchmark.json"))
    w("## 6. \u4e0e\u6ce8\u91ca\u534a\u72ec\u7acb\u7684\u7559\u51fa\u4f9b\u4f53\u57fa\u51c6\uff08\u6807\u5fd7\u57fa\u56e0\u9762\u677f\uff09\\n")
    w("- AT2 panel: " + ", ".join(M["panel_AT2"]))
    w("- \u6c14\u9053\u4e0a\u76ae/\u80bf\u7624 panel: " + ", ".join(M["panel_tumor"]))
    w(f"- \u540c\u4f9b\u4f53\u914d\u5bf9\u4f9b\u4f53\u6570: {M['n_paired_subjects']}\\n")
    for key, nm in [("i", "(i) GSE148071 \u8de8\u4f9b\u4f53\uff1aannotated malignant vs normal AT2"),
                    ("ii", "(ii) GSE131907 \u540c\u4f9b\u4f53\u5185\uff1a\u764c\u65c1\u4fa7 vs \u80bf\u7624\u4fa7\uff08\u53ea\u53d6 AT2 \u6ce8\u91ca\u7ec6\u80de\uff0clabel \u4e0d\u7528\u805a\u7c7b\u6ce8\u91ca\uff09")]:
        if key in M:
            w("**" + nm + "** \u2014 leave-one-donor-out AUROC")
            w("```"); w(pd.DataFrame(M[key]).to_string(index=False)); w("```")
    w("\n> \u53c2\u7167\u884c `[ref] marker panel` \u7ed9\u51fa\u300c\u539f\u59cb\u8868\u8fbe\u91cc\u6709\u591a\u5c11\u53ef\u8fc1\u79fb\u4fe1\u53f7\u300d\u7684\u6807\u5c3a\uff1b\u4e09\u4e2a embedding \u7a7a\u95f4\u5f7c\u6b64\u63a5\u8fd1 = \u5fae\u8c03\u5e76\u6ca1\u6709\u591a\u8bfb\u51fa\u751f\u7269\u5b66\u4fe1\u53f7\u3002\\n")

if os.path.exists(f"{RES}/paired_axis.json"):
    P = json.load(open(f"{RES}/paired_axis.json"))
    w("## 7. \u53ef\u8bc6\u522b\u6027\u4e0e\u540c\u4f9b\u4f53\u914d\u5bf9\u8f74\\n")
    w("```"); w(json.dumps(P.get("identifiability", {}), ensure_ascii=False, indent=1)); w("```")
    if "B_paired_axis" in P:
        w("\n\u764c\u65c1 AT2 \u2192 \u540c\u4f9b\u4f53\u80bf\u7624\u5185 AT2 \u7684\u914d\u5bf9\u65b9\u5411\uff08\u4f9b\u4f53\u6548\u5e94\u88ab\u5b8c\u5168\u6d88\u6389\uff09\uff1a")
        w("```"); w(pd.DataFrame(P["B_paired_axis"]).T.to_string()); w("```")
        w("\n> `mean_resultant_R` = \u5404\u4f9b\u4f53\u5dee\u5411\u91cf\u7684\u65b9\u5411\u4e00\u81f4\u6027\uff1b`LOPO_subject_rank_acc` = \u7528\u5176\u4f59\u4f9b\u4f53\u5b66\u5230\u7684\u8f74\uff0c\u5728**\u672a\u89c1\u4f9b\u4f53**\u4e0a\u628a\u4e24\u4fa7\u6392\u5bf9\u5e8f\u7684\u6bd4\u4f8b\uff08chance=0.5\uff09\u3002\\n")
    if os.path.exists(f"{RES}/donor_level_de_GSE148071.csv"):
        d = pd.read_csv(f"{RES}/donor_level_de_GSE148071.csv")
        w("**\u4f9b\u4f53\u7ea7 DE\uff08GSE148071\uff0c\u63a8\u65ad\u5355\u5143=\u4f9b\u4f53\uff09** \u2014 |t| \u524d 12\uff1a")
        w("```")
        w(d.reindex(d["welch_t"].abs().sort_values(ascending=False).index)
          .head(12)[["symbol", "mean_logFC", "welch_t", "p_welch", "q_welch"]].round(4).to_string(index=False))
        w("```")
        w("\n> \u5bf9\u7167\u4fa7\u53ea\u6709 5 \u4e2a\u4f9b\u4f53\uff0c\u4e14\u524d\u51e0\u540d\u662f HLA-DRA/HLA-DRB1/CYBA \u7b49 MHC-II/\u541e\u566c\u7ec6\u80de\u80cc\u666f\u57fa\u56e0\uff0c\u63d0\u793a\u8be5\u5bf9\u6bd4\u4ecd\u88ab\u7ec6\u80de\u7ec4\u6210\u4e0e\u4f9b\u4f53\u5dee\u5f02\u6c61\u67d3\uff0c\u4e0d\u80fd\u5f53\u6076\u6027\u6807\u5fd7\u7528\u3002\\n")

txt = "\n".join(md).replace("\\n", "\n")
open(f"{RES}/REPORT_zh.md", "w").write(txt + "\n")
print("\n".join(md[:30]).replace("\\n", "\n"))
print(f"\n... \u5199\u51fa {RES}/REPORT.md ({len(md)} \u884c)")
