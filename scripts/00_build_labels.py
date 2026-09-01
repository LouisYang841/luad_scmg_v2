#!/usr/bin/env python3
"""00_build_labels.py — 重建"干净"的 LUAD 细胞注释表。

动机（对上一轮工作的三处修正之一）：
  上一轮把 broad_cell_type=='Malignant' 当肿瘤细胞用，但该字段实际含义是
  "属于 Pure_Tumor 子图谱"，其中 58.5% 的 fine label 是 Normal Alveolar Epithelial。
  本脚本用 fine label 重新定义区室，并把 stage 与其真实嵌套结构（样本/组织部位/dataset）
  一起写清楚，供后续所有评估使用"样本级"而不是"细胞级"的推断单元。

输出:
  results/cell_annotations_clean.csv   (index = barcode, 与 zero-shot h5ad 顺序一致)
  results/label_audit.txt              (审计表)
"""
import os, re, sys
import numpy as np, pandas as pd

ROOT = "/home/ubuntu/luad_scmg_v2"
MAP = "/home/ubuntu/luad_invasion/results/metadata_cache/all_subatlases_barcode_to_label_mapping.csv"
OBS = "/tmp/luad_316k_scmg_umap_finetuned.csv"          # 定义 316,689 细胞集合与其顺序
OUT = f"{ROOT}/results"
os.makedirs(OUT, exist_ok=True)


def tissue_site(pid: str) -> str:
    """从 patient_id 推断组织部位（stage 与部位高度混淆，必须显式建模）。"""
    s = str(pid)
    if "LUNG_N" in s:
        return "normal_lung"
    if "LUNG_T" in s:
        return "primary_lung_tumor"
    if re.search(r"(_LN_|_EBUS_)", s):
        return "lymph_node"
    if "EFFUSION" in s:
        return "pleural_effusion"
    if re.search(r"_TD\d", s):
        return "primary_lung_tumor_section"
    if re.search(r"_P\d", s):
        return "primary_lung_tumor"
    return "other"


def main():
    lines = []
    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s); lines.append(s)

    obs = pd.read_csv(OBS, index_col=0, low_memory=False)
    mp = pd.read_csv(MAP)
    mp["cell_barcode"] = mp["cell_barcode"].astype(str)
    mp = mp.drop_duplicates("cell_barcode", keep="first")

    ann = obs[["broad_cell_type", "assigned_fine_label", "stage"]].copy()
    add = mp.set_index("cell_barcode").reindex(ann.index)
    ann["dataset"] = add["dataset"].values
    ann["patient_id"] = add["patient_id"].astype(str).values
    ann["sample_id"] = add["sample_id"].astype(str).values if "sample_id" in add else ann["patient_id"]
    ann["tissue_site"] = [tissue_site(p) for p in ann["patient_id"]]

    fine = ann["assigned_fine_label"].astype(str)
    # ---- 干净区室定义：只用 fine label，绝不使用子图谱成员身份 ----
    comp = np.where(fine.str.startswith("Tumor_"), "malignant_epithelium",
           np.where(fine == "Normal Alveolar Epithelial", "normal_alveolar_epithelial",
           np.where(fine == "NA", "unannotated", "other")))
    ann["compartment_clean"] = comp
    # 恶性细胞只在"肿瘤样本"里才可信；正常肺样本里出现的 Tumor_* 要单独标记
    ann["sample_is_lesional"] = (ann["stage"] != "Normal")
    ann["malignant_call"] = np.where(
        (comp == "malignant_epithelium") & ann["sample_is_lesional"], "malignant",
        np.where(comp == "malignant_epithelium", "malignant_called_in_normal_sample",
        np.where(comp == "normal_alveolar_epithelial", "normal_at2", "non_epithelial")))

    log("=" * 78)
    log("1) 上一轮 'Malignant' 区室的真实构成  (n = %d)" % (ann["broad_cell_type"] == "Malignant").sum())
    m = ann["broad_cell_type"] == "Malignant"
    log(ann.loc[m, "compartment_clean"].value_counts().to_string())
    log("  -> 'Malignant' 中真正的恶性上皮只占 %.1f%%"
        % (100 * (ann.loc[m, "compartment_clean"] == "malignant_epithelium").mean()))
    log("  -> 其中来自正常肺样本(stage==Normal)的细胞: %d"
        % ((m & (ann["stage"] == "Normal")).sum()))

    log("\n2) 干净区室统计")
    log(ann["compartment_clean"].value_counts().to_string())
    log("\n   malignant_call:")
    log(ann["malignant_call"].value_counts().to_string())

    log("\n3) 恶性上皮 / 正常AT2  × dataset")
    log(pd.crosstab(ann["malignant_call"], ann["dataset"]).to_string())

    log("\n3b) broad_cell_type × dataset（看子图谱身份本身如何随 dataset 变化）")
    log(pd.crosstab(ann["broad_cell_type"], ann["dataset"]).to_string())
    log("\n3c) Pure_Tumor 子图谱细胞的 fine label × dataset")
    mm = ann["broad_cell_type"] == "Malignant"
    log(pd.crosstab(ann.loc[mm, "compartment_clean"], ann.loc[mm, "dataset"]).to_string())

    log("\n4) '样本/供体'嵌套结构 —— stage 只是样本属性，且与部位、dataset 混淆")
    t = pd.DataFrame({
        "n_cells": ann.groupby("stage").size(),
        "n_patients": ann.groupby("stage")["patient_id"].nunique(),
        "n_datasets": ann.groupby("stage")["dataset"].nunique(),
        "tissue_sites": ann.groupby("stage")["tissue_site"].apply(lambda x: "|".join(sorted(set(x)))),
        "datasets": ann.groupby("stage")["dataset"].apply(lambda x: "|".join(sorted(set(x)))),
    })
    log(t.to_string())
    log("\n   ** AAH 只有一个样本；MIA 两个；AIS 三个 —— 细胞级的 stage '轨迹' 没有生物学重复。")

    log("\n5) 供后续评估用的对比组 (contrast groups)，推断单元 = patient_id")
    c1a = (ann["malignant_call"] == "malignant")
    c1b = (ann["compartment_clean"] == "normal_alveolar_epithelial") & (ann["stage"] == "Normal")
    c2b = (ann["compartment_clean"] == "normal_alveolar_epithelial") & (ann["stage"] != "Normal")
    for nm, mk in [("C1 恶性上皮(肿瘤样本)", c1a), ("C1b 正常肺AT2", c1b),
                   ("C2b 肿瘤样本内AT2", c2b)]:
        log("   %-22s n_cells=%-7d n_patients=%d n_datasets=%d"
            % (nm, mk.sum(), ann.loc[mk, "patient_id"].nunique(), ann.loc[mk, "dataset"].nunique()))
    ann["contrast_group"] = np.select(
        [c1a, c1b, c2b], ["malignant", "at2_normal_lung", "at2_in_tumor"], "other")

    # 细胞类型探针组（检查微调是否破坏全局细胞类型结构）
    ann["celltype_probe"] = np.where(ann["compartment_clean"] == "malignant_epithelium",
                                     "MalignantEpi",
                            np.where(ann["compartment_clean"] == "normal_alveolar_epithelial",
                                     "NormalAlvEpi", ann["broad_cell_type"]))

    ann.to_csv(f"{OUT}/cell_annotations_clean.csv")
    log("\n写出 %s/cell_annotations_clean.csv  shape=%s" % (OUT, ann.shape))
    with open(f"{OUT}/label_audit.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
