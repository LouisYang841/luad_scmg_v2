# 进度快照 / HANDOVER（2026-09-01 05:10 UTC）

给接手的人：**本文件是自足的**，不需要读聊天记录。权威报告是 `results/REPORT.md`（英文 9 节），
本文件只写"现在停在哪、下一步做什么、哪些结论还不能信"。

---

## 0. 2026-09-02 更新（CNV 阶段收尾 + 三项新证据）

接手者先读本节，§1-§8 为 09-01 旧快照（其中 §3 的阻塞已解除，见下）。

**§3 阻塞已解除**：按旧 §3 推荐方案 1 执行 —— 参考组用 (A) 记法的正常上皮，
阳性对照换成 GSE148071。自算 CNV（`71_infer_cnv.py`，W=100，CuPy）已跑通两数据集：
GSE148071 阳性对照 AUROC **0.7569**（malignant 13,412 vs held-out normal 4,026；
修复前的 0.395 记录在 `cnv/GSE148071/positive_control_metrics.json`，是旧版 bug 的残留）。

### 09-01 晚间至 09-02 新增的已验证结论（脚本 81-84，git c13a660+）

1. **novelty × CNV 双通道一致性呈层级结构**（`results/novelty_vs_cnv.json`）：
   细胞级硬判定 kappa≈0（148071 甚至 -0.01），但 **GSE131907 供体/样本级 Spearman 0.6225
   (p=2.3e-5)**，且部位梯度两法同序（adjacent 4.7%/1.2% → tumor 19%/10.7% → LN 46%/20.7% →
   胸水 56%/41%）。**梯度级结论（REPORT §9）被双正交证据钉死；细胞级对应关系不可主张。**
   另：148071 的 abn95 在两区室都标 ~60% 异常 —— novelty 阈值跨数据集不迁移（batch 效应），
   报告引用 novelty 时必须注明此局限。
2. **GSE131907 上皮恶性重建**（`scripts/83_cnv_malignant_recalls.py` →
   `cnv/GSE131907/epithelium_cnv_calls.csv`）：n=22,003，双证据恶性 2,220（10.1%），
   单证据 9,550，无证据 10,233；22/39 样本 CNV>5%，19/39 双证据>5%。
   注意 `patient_id=='GSE131907_Patient'` 是 9,521 细胞的元数据缺失桶（frac_cnv 0.49），
   不要当真供体。
3. **GSE131907 配对问题（旧 §8 待办 2）已用 GEO 官方元数据钉死**：
   `results/gse131907_geo_meta/GSE131907_samples.soft`（SOFT 原文）。
   10/10 对 LUNG_T##/LUNG_N## 的 `patient id` 完全一致（P0006…P0034），0 矛盾。
   **编号陷阱**：前缀语义不同 —— P####=原发、P1###=晚期/转移、P2###=正常淋巴结、
   P3###=脑转移；**跨前缀相同数字是巧合不是同一人**（EBUS_06=P1006 ≠ LUNG_T06=P0006）。
   同患者跨部位实例：EBUS_06+EFFUSION_06 同为 P1006。
   因此 81 的配对检验（10/10，p=0.00098）建立在已被官方元数据证实的配对上。
4. **Galaxy 官方 inferCNV 交叉验证**（`scripts/82_galaxy_vs_own_cnv.py` →
   `galaxy_vs_own_cnv.json`）：run #1 的 219.8MB "members" 输出实为逐细胞×16,937 基因
   ratio 矩阵；与自研 cnv_load 的 Spearman **0.598**，top-10% 异常细胞重叠 37%（随机 10%）。
   解释：基因集/归一化/参考集均不同 → 方向一致、实现各异，自研获部分背书。
   **run #1 没有正常对照**（8,016 个正常全进了 reference）→ 官方版 AUROC 缺位。
5. **Galaxy v2 已提交**（`scripts/84_galaxy_resubmit_with_normals.py`）：
   reference 只留 3,990 非 holdout 正常，4,026 held-out 正常移入 observations
   （与 71 的 0.7569 同定义，可直接对比）。history `luad_GSE148071_infercnv_v2`
   = 11ac94870d0bb33a9c8e05f5068ac355，job 4838ba20a6d86765caa987e8b1d01d98。
   完成后跑 `85`（待写）：下载 observations 矩阵算官方 malignant vs heldout AUROC。

### 接下来（按优先级）

1. Galaxy v2 完成后：下载 observations 矩阵 → 官方 AUROC vs 0.7569（`85`，待写）；
2. 把 §0 结论并入 `49_report_en.py`（注意 49 会整篇重写 REPORT.md，须改脚本不改成品）；
3. 旧 §8 剩余待办：5 张 GSE189487 切片 stage 纠错（GEO 原始元数据可同样抓取）；
   GSE131907 "Patient" 缺失桶的样本归属。



---

## 1. 一句话状态

微调与生态位两阶段已完成并入报告；**CNV 自算这一阶段刚开始、还没有任何结果**，
卡在"参考细胞怎么定"这个未解决的数据冲突上（见 §3，必须先解决再跑）。

## 2. 已完成、已验证、已提交（git）

| 阶段 | 脚本 | 产出 | 状态 |
|---|---|---|---|
| 干净注释 + 标签审计 | `00_build_labels.py` | `results/cell_annotations_clean.csv`, `label_audit.txt` | ✅ |
| 标准矩阵落盘 | `10_prep_gpu_matrix.py` | `data/X_std_18108.f16` + `cell_order.csv` | ✅ 316,689×18,108 fp16 |
| 无监督适配（官方 loss、同数据集负样本、供体留出） | `20_finetune_unsup.py` | `results/finetune_unsup/`，**用 `chosen_state.pth`(=epoch1)** | ✅ 23 min/L4 |
| 上一轮可复现审计 | `25_audit_prev_work.py` | `results/audit_prev_work.json` | ✅ |
| 评估台（掩码恢复选型 + CKA/kNN 保持） | `30_eval_harness.py` | `sweep_metrics.csv`, `comparison_models_final.csv` | ✅ |
| 可识别性 + 同供体配对轴 + 供体级 DE | `35_paired_malignancy_axis.py` | `paired_axis.json`, `donor_level_de_*.csv` | ✅ |
| 不依赖注释的留出供体基准 | `37_marker_benchmark.py` | `marker_benchmark.json` | ✅ |
| 模型原生映射（官方 CellTypeSearcher 语义） | `60/61/62` | `results/model_native/` | ✅ |
| 空转生态位重做 + 新旧对比 + 消融 | `63/65` | `results/niches/` | ✅ |
| 报告 | `49_report_en.py`（最后跑） | `results/REPORT.md` | ✅ |

环境：`/home/ubuntu/qwen-env/bin/python`，L4 23 GB，磁盘剩 113 GB。
`run_all.sh` 与 `README.md` 已同步（README 有逐脚本 I/O 表、外部依赖表、10 条结论速查、待办、已知坑）。

## 3. ⚠️ 阻塞项：两套"正常上皮"标签互相矛盾，CNV 参考组无法定

今晚查朋友新传的文件时发现的，**这正是他说的"某个数据集编号在合并时丢失、被并成另一个"**。

同一批 316,689 细胞里，"正常肺泡上皮"的总量有两个互不相容的答案：

| 来源 | 记法 | 正常上皮细胞数 | 按数据集 |
|---|---|---|---|
| **(A)** `all_subatlases_barcode_to_label_mapping.csv` 的 `major_compartment=='Normal_Epithelial'`（= 我们的 `compartment_clean=='normal_alveolar_epithelial'`） | 子图谱区室 | **31,366** | GSE131907 23,350 / GSE148071 8,016 / **GSE189357 = 0** |
| **(B)** `cell_metadata_by_sample.csv` 的 marker 判 `celltype`（= `cell_order.malignant_call=='normal_at2'`） | 人工 marker | **73,945** | GSE131907 26,342 / GSE189357 15,462 / GSE148071 32,141 |

顺带确认的硬事实：**整个 atlas 的 8 个区室里根本没有气道上皮**
（T_cell 130,491 / Myeloid 66,347 / B_cell 31,476 / Normal_Epithelial 31,366 / Pure_Tumor 22,288 /
Plasma 17,738 / Fibroblast 10,326 / Endothelial 6,657）——所以连"支气管上皮=整倍体"这条 inferCNV 常规参考路都被堵死。

**为什么这直接卡住 CNV**：inferCNV 类方法的性质由参考组完全决定。(A) 与 (B) 差 42,579 个细胞，
且 **GSE189357 在 (A) 下没有正常上皮 → 该数据集无法建参考**，而它是"已知有明确恶性上皮"的阳性对照数据集。

### 接手后要做的决定（我推荐 1）
1. **用 (A)**，并把阳性对照换成 GSE148071（13,412 annotated malignant vs 8,016 normal，同数据集）；
   GSE131907 参考只能用 23,350 个 AT2 里的**癌旁部位部分**（仅 2,743 个 → 偏小，且癌旁 AT2 是否干净本身待检）。
   代价：参考小、需在报告里明写。
2. 用 (B)，但 (B) 是人工 marker 判的、含 `SFTPC+/PDPN−` 规则，可靠性未核。
3. 先问朋友 (A)/(B) 哪个是他最终采用的、另一个是哪一步的残留 —— **最省时间，建议先问**。

## 4. CNV 阶段已完成的部分（不要重做）

- ✅ GENCODE v44 basic GTF 已下载解压：`ref/gencode.v44.basic.annotation.gtf.gz`（29 MB，26 s）
- ✅ `scripts/70_gene_positions.py` → `results/gene_positions_gencode44.csv`
  （62,663 个 gene 记录，其中 **protein_coding 20,033**；标准 18,108 空间内落在 chr1-22,X 的有 **16,976** 个可用基因）
  含 `ensg, chrom, start, end, gene_type, arm, centromere`；臂用硬编码 GRCh38 着丝粒中点分 p/q。
- ✅ `scripts/71_infer_cnv.py` 已写好并修掉两个 bug（臂汇总索引写坏 → 改成 `np.add.at` 分段聚合；
  参考站点筛空 → 改成从 `patient_id` 解析 `adjacent_normal/tumor_sample/lymph_node/pleural_effusion/airway`）。
  **设计要点（已实现）**：留一半正常细胞当"同状态诚实零分布"定阈值（不用参考细胞自己定阈值，避免循环收紧）。
- ❌ **还没有跑出任何结果**。曾经死在 `cp.load('data/X_std_18108.f16')`（pickled-object 报错），
  已改为 `L.load_X_std()`。**现在唯一的阻塞是 §3 的参考组定义**，不是代码。

### 跑之前必查的两行输出
```
GSE189357: ref=0 ...      <- 必须 >0，否则该数据集没参考，阳性对照做不了
```
判定方法是否可信的唯一标准：**在 GSE148071 / GSE189357 上，annotated malignant 与 annotated normal 上皮
要能分开（AUROC 明显 >0.7）**。分不开就说明 10x 3' 低复杂度数据（nFeature 中位仅 ~1,100）算不出可靠 CNV，
那整条 CNV 路线应放弃，改为继续依赖 §9 的 novelty 判据。

## 5. 今晚新查出的其他事实（已核实）

- 朋友传的 `super_atlas_300k_qc_passed.rds`：33,968 基因 × **297,954** 细胞，
  mtime **7/29 08:50**，由 `34_super_atlas_harmony.R` **在注释之前**保存；
  只有 2 个数据集（**无 GSE148071**）、6 个 meta 列、**无任何注释/降维/CNV 列**、基因是 **symbol**（0/33,968 ENSG）。
  counts 本身干净：全整数、总 UMI 1.73e9、`colSums(counts)==nCount_RNA` 逐细胞成立、barcode 无重复、QC 阈值确实执行过
  （nCount [1000,59961]、nFeature [500,8934]、pct.mt [0,9.999]）。
  → **它不能解决任何问题**，要请他改传"带注释 + 带 CNV 列"的最终对象，或直接给 CNV 表。
  唯一可用之处：未做 Harmony 的 raw counts。
- 该对象里 **GSE131907 全部 192,665 细胞的 `sample_id` 是 NA**，供体只剩细胞名后缀；`percent.mt` 只在 **13 个 MT 基因**上算
  → `<10%` 比字面更宽松，而降解细胞恰以 AT2 最敏感。
- 我自己的三处输入已逐细胞核对完全一致（`h5ad.obs` / `cell_order.csv` / `cell_annotations_clean.csv`，0 不一致），
  且 barcode 命名空间与 dataset 标签无交叉 → **他说的"编号丢失"没有污染我今晚用的输入**。
  证据链：embedding 行序与 h5ad obs_names 逐位相同（316,689）；留出供体细胞类型准确率 0.93（错位只会是 1/6）。
- 上游 `metadata_cache/` 里 `_upgraded.csv` 名字比 `mapping.csv` 更"高级"但**时间更早**（8/25 vs 8/31），
  且是 `mapping.csv` 的真子集（少 31,366 = 正好 Normal_Epithelial 区室全体）→ 他需要知道的又一处新旧混淆。

## 6. 明确不能信的东西

1. **CNV 的任何数字** —— 一行都还没跑出来。谁先看到 `results/cnv/` 里有东西，先确认它是在 §3 决定之后生成的。
2. 上一轮的 `Tumor_C0..C13`「克隆」、`broad_cell_type=='Malignant'`、以及用它们算出的 ARI/轨迹分期/克隆-分期 Fisher 检验。
3. 上一轮的 6 个 Visium「生态位」当独立发现（它们是手写线性 score）。
4. 旧空间表里 15 张切片中 5 张的 `stage`（与 ground-truth manifest 矛盾，全在 GSE189487）。
5. `all_slides_spatial_deconv_master.csv`（只含 7/15 张）。
6. GSE131907 的 `LUNG_T##/LUNG_N##` 同供体配对**仍是命名推断**，未经 GEO 核实；
   REPORT §9 的 9/9 p=0.004 建立在此之上。若 GEO 不成立，该结论降级为"未配对的中位数差异"。

## 7. 下一步（按优先级）

1. 问朋友 §3 的 (A)/(B) 哪个权威（或直接把两个都给他看让他认）。
2. 改 `71` 一行 `cp.load` → `L.load_X_std()`，跑通，**只看阳性对照**。
3. 阳性对照过 → 出 GSE131907 的 per-cell CNV，与 §9 的 `abn95` 算 AUROC + 同供体配对 → 写进 REPORT §10。
4. 阳性对照不过 → 在 REPORT 里明写"该数据不支持可靠的表达推断 CNV"，把结论锁在 novelty 判据上。
5. 拿 CNV 定义重新构造对比组 → 按 `README §0` 的原则做**第二次微调**（这次标签来自测量而非人工注释），
   仍用留出供体评估。这一步是朋友最初想做但做歪的事的正确版本。
6. 请他核实 GEO 配对（§6.6）。

## 8. 复现最小命令

```bash
cd /home/ubuntu/luad_scmg_v2
bash scripts/relink_tmp.sh                       # /tmp 易失，先恢复上一轮产物软链
bash run_all.sh                                  # 全流程；49 必须最后
# CNV 阶段（当前不可跑，见 §3/§4）
/home/ubuntu/qwen-env/bin/python scripts/70_gene_positions.py
CNV_W=100 /home/ubuntu/qwen-env/bin/python scripts/71_infer_cnv.py
```
