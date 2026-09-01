# luad_scmg_v2 — LUAD × SCMG 返工工作区（交接文档）

对 2026-08-31 晚间那一轮 SCMG 微调（脚本散落在 `/tmp/`，产物在 `/tmp/scmg_*`）的**方法学返工 + 下游重做**。
原目录 `/home/ubuntu/luad_invasion`（上游图谱/空转结果）与 `/home/ubuntu/scmg_workspace`（SCMG 代码与官方权重）**只读，未修改**。

## 0. 先看这里：该用哪个权重

| 用途 | 路径 | 说明 |
|---|---|---|
| **LUAD 适配版（推荐）** | `results/finetune_unsup/chosen_state.pth` | = `state_epoch1.pth`。无监督、官方三项损失、全 316k、供体级留出；留出供体上三项指标最好 |
| 原始全局模型（默认就该用这个） | `/home/ubuntu/scmg_workspace/models/embedder/` | SCMG 官方 zero-shot embedder，3.2 GB。**绝大多数分析不需要微调** |
| ❌ 不要用 | `/tmp/scmg_luad_finetuned_opt/model.pt` | 上一轮 broad-label hard-negative 版（标签泄漏） |
| ❌ 不要用 | `/tmp/scmg_stage_ft/stage_model.pt` | 上一轮 stage-aware 版（标签泄漏 + 只在 53k 上训） |
| ❌ 不要用 | `/tmp/diffusion_luad_stage/` | 上一轮扩散模型（欠训练，生成的"轨迹"是连线假象） |

结论依据：`results/REPORT.md` 第 2、4、5、6 节。一句话——**zero-shot 已携带几乎全部可迁移的恶性/去分化信号，微调 encoder 没有收益；唯一有增益的是 dataset token/decoder 适配，且必须限制在 1 个 epoch。**

## 1. 目录结构

```
luad_scmg_v2/
├── README.md                  本文件
├── run_all.sh                 全流程复现（按序执行）
├── scripts/                   代码（编号即执行顺序）
├── data/                      大中间文件（可重建，不入版本库）
│   ├── inputs_previous_round/ 上一轮 /tmp 产物的归档（硬链接，防 /tmp 被清）
│   └── X_std_18108.f16        316,689 × 18,108 标准矩阵，11.5 GB
├── ref/                       SCMG 官方全局参考图谱（从 HF 下载，3.0 GB）
├── results/                   所有产出（表格 / json / 图 / 权重）
└── logs/                      每次运行的日志
```

## 2. scripts/ 逐个说明

公共库先列：

| 文件 | 作用 |
|---|---|
| `luadlib.py` | 共用函数：显存矩阵加载、模型加载（含 state_dict 的 token 名回填）、`add_dataset_tokens`（给新数据集加 decoder 条件 token）、官方 loss、faiss/cuml kNN、**linear CKA / kNN 图重叠 / effective rank / leave-one-patient-out 探针** |
| `relink_tmp.sh` | **`/tmp` 是易失的**。本脚本从 `data/inputs_previous_round/` 把上一轮产物重新链回 `/tmp`，使 25/30/35/37/45/63 可复现。机器重启后先跑这个 |

### A. 数据准备

| 文件 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `00_build_labels.py` | luad_invasion 的注释表 | `results/cell_annotations_clean.csv`, `label_audit.txt` | 用 **fine label** 重建干净区室（`malignant_epithelium` / `normal_alveolar_epithelial`），并审计 `broad_cell_type=='Malignant'` 的真实构成与 stage 的嵌套混淆 |
| `10_prep_gpu_matrix.py` | `/tmp/luad_316k_scmg_ready.h5ad` | `data/X_std_18108.f16`, `cell_order.csv`, `std_genes.npy`, `prep_config.json` | 一次性把 316,689 细胞转成 SCMG 标准 18,108 维（CPM 1e4 + log1p），与官方 `get_Xs_from_anndata` 逐位一致；fp16 落盘供显存直载 |

### B. 模型训练与审计

| 文件 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `20_finetune_unsup.py` | `data/X_std_18108.f16` + 官方权重 | `results/finetune_unsup/{history.json, best_meta.json, state_epoch*.pth, chosen_state.pth, emb_val_last.npy}` | **诚实版微调**：官方三项 loss（contrastive×1000 + l2×0.1 + **recon×1e-3**）、负样本=同 dataset 随机（官方规则）、kNN 图每 3 epoch 重建、3 个 GSE 新建 token（预训练 37 行冻结）、全 316k 细胞、**留出 15 个完整供体**。~23 min / L4。`--smoke` 快速自检 |
| `25_audit_prev_work.py` | `/tmp` 旧 embedding/权重 + `diffusion_transition_100.npy` | `results/audit_prev_work.json` | 把对上一轮的审查变成可复现数字：stage 探针（未见供体 vs 随机切分）、扩散轨迹相邻/任意距离比、off-manifold、recon 一致性、CKA。低显存，可与训练并发 |

### C. 评估

| 文件 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `30_eval_harness.py` | `data/` + 各 checkpoint | `results/sweep_metrics.csv`（`--mode sweep`）, `comparison_models*.csv` + `emb_val_<tag>.npy`（`--mode final`） | 评估台。选型用 **held-out masked-gene recovery**（label-free 且不退化）；`val_c` 只作为"偏离全局流形多少"报告，**不用于选型**。另含 LOPO 探针、CKA、kNN 重叠、effective rank。`--specs "tag=path;..."` 指定要比的模型 |
| `35_paired_malignancy_axis.py` | `data/` + embedding | `results/paired_axis.json`, `donor_level_de_GSE148071.csv` | 供体级可识别性（同一 patient_id 内两类细胞共存的供体 = 0）+ **同供体配对恶性轴**（R、LOPO 排序准确率）+ 供体级 pseudobulk DE（推断单元=供体，不是细胞） |
| `37_marker_benchmark.py` | `data/` + embedding + 参考 marker 表 | `results/marker_benchmark.{csv,json}` | 与聚类注释**半独立**的留出供体基准：(i) GSE148071 跨供体恶性 vs 正常AT2；(ii) GSE131907 同供体癌旁 vs 肿瘤侧（label 不用聚类注释）。附 marker 面板参照 AUROC |
| `45_summary_figure.py` | 上述 csv | `results/summary_rework.png` | 4 联图：适应/流形保持权衡、循环评估的来源、扩散轨迹假象、多空间对比 |

### D. 模型自带映射（不用聚类标签）

| 文件 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `60_model_native_annot.py` | `ref/…manifold.h5ad` + **冻结** zero-shot embedding | `results/model_native/cell_native_annotations.csv`, `annot_summary.json` | SCMG 官方 `CellTypeSearcher` 语义：对全局参考（133,061 细胞 / 797 类型 / 30 major）做 kNN(k=25) + `exp(-(d/2)²)` 软权重迁移，给每个细胞模型原生身份 + 到参考的最近邻距离。**必须用冻结模型**——参考向量存在于 zero-shot 空间里，换空间这张地图就失效 |
| `61_epithelium_novelty.py` | `ref/` + `data/` | `model_native/epithelium_novelty.{csv,json}` | 裸 novelty 被参考类别密度支配（浆细胞比多数肿瘤 clone 还"新奇"）→ 改成**只在上皮参考里找最近邻**。核心输出：GSE131907 同供体配对检验 |
| `62_epithelium_calls.py` | 同上 | `epithelium_thresholds.json`, `epithelium_calls.csv`, `donor_epithelium_calls.csv` | 用参考自身 1NN 距离做零分布 + 癌旁正常上皮 q95 阈值，产出 `abn95` 异常上皮判定，并按采集部位/供体汇总 |

### E. 空转生态位

| 文件 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `63_model_native_niches.py` | 旧去卷积表（15 张单切片 csv）+ `M` 交叉表 + ground-truth manifest | `niches/{spot_niches.csv, archetype_definitions.csv, old_vs_new_spearman.csv, slide_x_*.csv, patient_x_niche_native.csv, within_patient_pre_vs_IAC.csv, niche_summary.json}` | 用模型原生身份重做生态位：旧 30 个去卷积标签 → 16 个原生身份的**交叉表重映射**；切片内 k=6 + 跨切片原型匹配；**消融实验**（同一算法只把上皮换回旧标签 → ARI）。注意本机无空转原始矩阵，所以这是标签重映射，不是重新去卷积 |
| `65_niche_figure.py` | 上述 | `niches/niche_comparison.png` | 生态位对比三联图（原生原型定义 / 旧 score vs 原生成分 Spearman / 每个旧 score 最强锚点） |

### F. 报告

| 文件 | 输出 | 作用 |
|---|---|---|
| `49_report_en.py` | **`results/REPORT.md`（权威版本，英文）** | 汇总 1–9 节。**放在最后执行**，它会整篇重写 REPORT.md |
| `40_report.py` | `results/REPORT_zh.md`（中文，旧版，不含第 8–9 节） | 保留备查 |

> 约定：**生成脚本里的文本一律用英文**。中文写在 shell heredoc 里会被转义反复咬到。

## 3. data/ 与 ref/

| 文件 | 大小 | 说明 | 可重建 |
|---|---|---|---|
| `data/X_std_18108.f16` | 11.5 GB | 316,689×18,108 标准矩阵（fp16）。所有训练/评估的地基 | 是（`10_prep_gpu_matrix.py`，128 s） |
| `data/cell_order.csv` | 33 MB | **行顺序即矩阵行顺序**：barcode / dataset / patient_id / tissue_site / stage / malignant_call / celltype_probe | 是 |
| `data/std_genes.npy`, `prep_config.json` | — | 18,108 标准基因 ENSG + 预处理元信息 | 是 |
| `data/inputs_previous_round/` | 7.1 GB（硬链接） | 上一轮 `/tmp` 产物归档（h5ad、两个 embedding、两个 leaky 权重、扩散样本、stage_umap）。配 `scripts/relink_tmp.sh` | **不可重建**，唯一副本 |
| `ref/data/ref_global_cell_state_manifold.h5ad` | 3.0 GB | SCMG 官方全局参考：133,061 细胞 × 18,108，`obsm['X_scmg']` + 797 `cell_type` / 30 `major_cell_type` / 331 tissue；含 11,501 个肺细胞；**无任何肿瘤数据集** | 是（HF：`xingjiepan/SCMG_data`，18 s） |

## 4. results/ 产物索引

| 文件 | 内容 |
|---|---|
| **`REPORT.md`** | 权威报告（英文，9 节，含所有表格与结论） |
| `REPORT_zh.md` | 中文旧版（1–7 节） |
| `cell_annotations_clean.csv` | 53 MB，316,689 行的干净注释表（下游一切的输入） |
| `label_audit.txt` | 标签审计原始输出 |
| `audit_prev_work.json` | 对上一轮的全部可复现审计数字 |
| `finetune_unsup/history.json` | 15 epoch 的训练/验证曲线（train_c, val_c, val_recon_corr, eff_rank, mean_norm） |
| `finetune_unsup/best_meta.json` | 留出供体清单（15 个）、超参、best_epoch |
| `finetune_unsup/state_epoch*.pth` | 每 epoch 的 state_dict（`best_*` = epoch1，因 val_loss 选型退化；`chosen_state.pth` 才是按正确指标选的） |
| `sweep_metrics.csv` | ZS + 15 个 epoch 的留出供体指标（掩码恢复、recon、mean_norm） |
| `comparison_models_final.csv` | 5 个空间正面比较（`comparison_models.csv` 是其副本） |
| `emb_val_<tag>.npy` | 5×150 MB，75,809 个留出供体细胞的 embedding（画图/复核用） |
| `marker_benchmark.{json,csv}` | 两个留出供体基准的 AUROC |
| `paired_axis.json` | 可识别性数字 + 同供体配对轴 |
| `donor_level_de_GSE148071.csv` | 供体级 DE（18,108 基因，含 Welch t / p / q） |
| `model_native/cell_native_annotations.csv` | 65 MB，316k 细胞的模型原生身份 + 到参考距离 |
| `model_native/epithelium_calls.csv` | 53,654 个上皮细胞的异常度与 `abn95` 判定 |
| `model_native/donor_epithelium_calls.csv` | 供体级汇总 |
| `niches/spot_niches.csv` | 118,565 个 spot：slide / stage / 供体 / 新原生生态位原型 / 消融版标签 |
| `niches/old_vs_new_spearman.csv` | 旧 6 手写 score × (8 原型 + 16 原生成分) 的切片级 Spearman |
| `niches/within_patient_pre_vs_IAC.csv` | n=3 供体配对：癌前 vs IAC 的成分类差值 |
| `summary_rework.png`, `niches/niche_comparison.png` | 两张图 |
| `logs/*.log` | prep / finetune_unsup / sweep / final / paired / marker / niches / hf_ref |

## 5. 外部依赖（绝对路径）

| 路径 | 用途 | 读写 |
|---|---|---|
| `/home/ubuntu/scmg_workspace/SCMG` | SCMG 源码（`sys.path.insert` 引入）。`manifold_generation.py` 有一处**修上游 bug** 的 patch：`train_diffusion_model` 把 `cond_classes` 字符串直接传给 `backward()`，而它需要 condition embedding | 只读（含 1 处已提交改动） |
| `/home/ubuntu/scmg_workspace/models/embedder` | 官方 zero-shot 权重 | 只读 |
| `/home/ubuntu/luad_invasion/results/…` | 上游图谱注释表、空转丰度表、ground-truth manifest | 只读 |
| `/home/ubuntu/luad_invasion/pipeline/` | 上游脚本（`build_11_clones.R:21` 是 "Tumor C" = cluster 改名的出处；`12_…_niches.py` 的 6 个"生态位"是手写线性 score） | 只读 |
| `/home/ubuntu/luad_invasion/results/sub_atlases/Pure_Tumor_subatlas.rds` | 含第二套 stage 列 `stage_true`（与 mapping 表冲突） | 只读 |
| `/tmp/scmg_*`, `/tmp/luad_316k_*` | 上一轮产物。**易失**，已归档到 `data/inputs_previous_round/` | 只读 |
| `/home/ubuntu/qwen-env/bin/python` | 运行环境 | — |

## 6. 环境与资源

- **GPU**：NVIDIA L4 23 GB。训练峰值 ~19 GB（`X_all` 常驻显存 11.5 GB fp16），88 s/epoch。
- **RAM** 47 GB；**磁盘**：根分区 193 GB，本项目占 ~21 GB。
- Python：`/home/ubuntu/qwen-env/bin/python`（torch 2.13+cu130、faiss 1.15、cuml 26.08、cupy、scanpy、anndata、pandas 3.0）。
- 版本库：本目录已 `git init`，`.gitignore` 排除 `data/ ref/ results/ logs/`（体积过大），**只跟踪代码与 README**。

## 7. 关键结论速查（细节见 REPORT.md）

1. `broad_cell_type=='Malignant'` 是"Pure_Tumor 子图谱成员身份"，58.5% 是正常 AT2 注释，且含义随 dataset 漂移 → 上一轮的负样本构造踩在这个字段上。
2. `stage` 是样本级属性，与 dataset/组织部位完全混淆；且 mapping 表的 stage 丢失了 `stage_true` 里的信息。
3. 用同一批标签训练又用它评估 → 循环。换成未见供体：LOPO bAcc 0.369 / 0.402 / 0.401（chance 0.167），而随机切分 0.76–0.79。
4. 扩散"Normal→IAC 轨迹"是 100 个独立采样点连线的假象（相邻/任意距离比 0.97，off-manifold 2.33 vs 0.84）。
5. 正确指标下微调也**没有**读出更多生物信号：留出供体 AUROC 三空间打平（0.814 / 0.822 / 0.816；同供体设计 0.928–0.934）。
6. 收益只在 epoch 1，之后过拟合供体（掩码恢复 0.657 → 0.612，低于 zero-shot 的 0.627；kNN 重叠 0.69 → 0.45）。
7. 模型原生映射（不碰聚类标签）下，旧 6 生态位里 4 个方向一致可复现；`Exhausted_TME` 实为"正常上皮少"的同义词；`Invasive_Solid_Core` 锚在"无正常肺对应物的上皮"。
8. 两个必须先修的数据缺陷：master 去卷积表只含 7/15 张切片；**15 张里 5 张 stage 与 ground-truth manifest 矛盾**（全在 GSE189487）。
9. 不依赖注释也不依赖 CNV 的上皮异常度判据：**9/9 供体**、p=0.004 地证明 GSE131907 肿瘤样本里的上皮确实异常 → "0 个恶性上皮"是注释错误。癌旁 5% → 原发 19% → 淋巴结转移 46% → 胸水 56%。
10. 供体内配对（GSE307534 的 P1–P3，n=3）：没有任何生态位从癌前到侵袭发生可重复改变。**变的是上皮细胞本身，不是它的生态位。**

## 8. 待办 / 需要对方提供

- [ ] **CNV 表**（`barcode, cnv_score, cnv_call, method`，另加：工具、参考细胞怎么定、分辨率 bin/arm/genomewide）。到手即可算 novelty vs CNV 的 AUROC，把第 9 节钉死。
- [ ] 核实 GSE131907 `LUNG_T##` / `LUNG_N##` 是否同供体（本轮所有配对检验都建立在这个命名推断上；GEO 页面抓取失败）。不成立则第 9 节与 `within_patient_pre_vs_IAC.csv` 需重做。
- [ ] 修正 5 张切片的 stage 归属；把 15 张切片重新并入 master。
- [ ] 用 CNV 重做 GSE131907 的上皮恶性判定（现有 23,350 个细胞的 fine label 全错）。
- [ ] 若要真·重去卷积：需要空转原始矩阵（本机没有，只有丰度表和坐标）。
- [ ] 扩散/生成未重训——等 encoder 与标签定义定下来再说。

## 9. 已知坑

1. **`/tmp` 易失**：25/30/35/37/45/63 有硬编码 `/tmp` 路径。机器重启后先跑 `bash scripts/relink_tmp.sh`。
2. **报告生成顺序**：`49_report_en.py` 必须最后跑（它整篇重写 `REPORT.md`）。
3. 脚本内文本用英文（见 §2F 约定）。
4. `results/summary_rework.png` 与 `niches/niche_comparison.png` **未经人工目视核验**（当前模型不支持图像输入），接手时请先看一眼版式与坐标轴。
5. 训练脚本只固定了 `torch.manual_seed` + `np.random.seed`；faiss GPU / TF32 / bf16 路径下不保证逐位可复现（指标可比，小数点末位不可比）。
