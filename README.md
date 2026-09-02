# luad_scmg_v2 — LUAD × SCMG 返工工作区（交接文档）

对 2026-08-31 晚间那一轮 SCMG 微调（脚本散落在 `/tmp/`，产物在 `/tmp/scmg_*`）的**方法学返工 + 下游重做**。
原目录 `/home/ubuntu/luad_invasion`（上游图谱/空转结果）与 `/home/ubuntu/scmg_workspace`（SCMG 代码与官方权重）**只读，未修改**。

> **先看 [`HANDOVER.md`](HANDOVER.md)** —— 进度快照：什么已验证、什么正卡住、什么结论还不能信。
> 权威分析结果在 [`results/REPORT.md`](results/REPORT.md)（英文 9 节）。

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
- [x] 核实 GSE131907 `LUNG_T##` / `LUNG_N##` 是否同供体 → **已用 GEO 官方 SOFT 钉死**：10/10 对 `patient id` 完全一致（P0006…P0034），0 矛盾。证据 `results/gse131907_geo_meta/GSE131907_samples.soft`。
  注意前缀语义不同：**P#### 原发 / P1### 晚期转移 / P2### 正常淋巴结 / P3### 脑转移**，跨前缀相同数字是巧合不是同一人（EBUS_06=P1006 ≠ LUNG_T06=P0006）。
- [ ] 修正 5 张切片的 stage 归属；把 15 张切片重新并入 master（GSE189487 的 stage 可用同样方式从 GEO 抓）。
- [ ] 用 CNV 重做 GSE131907 的上皮恶性判定（现有 23,350 个细胞的 fine label 全错）→ 已部分完成：`cnv/GSE131907/epithelium_cnv_calls.csv`（22,003 个上皮，双证据恶性 2,220）。
- [ ] 若要真·重去卷积：需要空转原始矩阵（本机没有，只有丰度表和坐标）。
- [ ] 扩散/生成未重训——等 encoder 与标签定义定下来再说。

**2026-09-02 新增（大方向相关，背景见 §10）：**

- [x] **核实竞争文献**（2026-09-02 完成，用 NCBI E-utilities + Crossref 逐条核对，结果见 §10.5 / §10.9）。
- [ ] 判定 CNV 能否充当"外部锚定真值"：自算 AUROC 0.7569（GSE148071）对做下游标签勉强够，**对做 benchmark 真值不够**。见 §10.3。**（现在是最高优先——st2traj 已证明 agreement 路线被占，外部锚是唯一剩下的可发表性来源。）**
- [ ] 定义 CNV 谱系方向性（共享 CNV 事件 → 祖先/衍生），评估其作为轨迹方向外部锚的可行性（§10.3）。
- [ ] 用 GSE189487 的 15 张 Visium 切片校验合成 spot 的**组成统计量**是否与真实 Visium 匹配（§10.4.3）。
- [ ] 定最终研究问题措辞：二元"可不可靠" → "在哪个颗粒度上可靠"（§10.2）。
- [ ] Galaxy v2 官方 AUROC（`85`）：**不主动推进**。86 重试循环已停（`logs/86_retry_loop.log` 仅 1 次 attempt，error），判定为 Galaxy EU 平台端故障。**不要反复重提交**，等平台恢复或有别的官方 CNV 来源再说。

## 9. 已知坑

1. **`/tmp` 易失**：25/30/35/37/45/63 有硬编码 `/tmp` 路径。机器重启后先跑 `bash scripts/relink_tmp.sh`。
2. **报告生成顺序**：`49_report_en.py` 必须最后跑（它整篇重写 `REPORT.md`）。
3. 脚本内文本用英文（见 §2F 约定）。
4. `results/summary_rework.png` 与 `niches/niche_comparison.png` **未经人工目视核验**（当前模型不支持图像输入），接手时请先看一眼版式与坐标轴。
5. **权威细胞数**：GSE131907 183,736 / GSE189357 95,624 / GSE148071 37,329，合计 316,689。
   已在 `h5ad.obs`、`data/cell_order.csv`、`results/cell_annotations_clean.csv` 三处逐细胞核对为 0 不一致。
   交接时曾有一处口头结论误用旧分组，把 GSE148071 说成"0 个恶性上皮"——实际是
   21,428 个 Malignant 桶细胞里 13,412 个为恶性上皮、8,016 个不是（见 REPORT §1b）。**别照抄旧消息。**
6. 训练脚本只固定了 `torch.manual_seed` + `np.random.seed`；faiss GPU / TF32 / bf16 路径下不保证逐位可复现（指标可比，小数点末位不可比）。

## 10. 大方向：解卷积 → 细胞状态空间 → 轨迹/药物的可靠性基准

> 本节记录 2026-09-02 的战略讨论，**不是已验证结论**。
> 来源是方向笔记 `D:\Obsidian-Note-Vault\Untitled 1.md`（实为一段保存下来的 AI 对话记录，非定稿计划）及其批判性审阅。
> 写 proposal 前必须先处理 §10.4–10.6。

### 10.1 我们在大方向里的位置

目标链路：`空间 spot 表达 → 解卷积 → 细胞组成/状态 → scMGCA / scGPT 类模型 → 轨迹与状态转移 → 药物优先级`。

两端都已有工具（解卷积有 benchmark；药物预测有 SpaRx、OSCC TC→LE 论文做模板）。**真正的空白在中间的跨层级接口**：`spot-level composition` ≠ `cell-level expression`，把它当单细胞表达喂给下游模型这件事缺少端到端验证。

我们目前**不在核心实验里，而在它前面——我们在造"真值分支"**。笔记的整个设计依赖"真实分支必须真的是真的"这个前提，而手上的活正好是这件事：

| 大方向的位置 | 我们已做 / 在做 | 脚本 |
|---|---|---|
| 真值：可信的细胞级恶性 / 状态定义 | CNV 作为**测量级**标签，替代人工注释 | 71 / 82 / 83 / 85 |
| 第二层：细胞状态表示是否可信 | SCMG zero-shot vs 微调、CKA / kNN 保持、留出供体 | 20 / 30 / 35 / 37 |
| 第一层：空间定位（诚实做法 = spot 级组成状态，而非伪细胞） | Visium 生态位重做 63/65。**注意上游 6 个"生态位"是手写线性 score**（`/home/ubuntu/luad_invasion/pipeline/12_…_niches.py`），正是本节要避免的陷阱 | 63 / 65 |
| 误差归因的前置 | GEO 官方元数据钉死同供体配对 | 82 / 83 |

### 10.2 核心问题的措辞（关键）

不要问"解卷积结果可靠吗"——二元、且没有分母。建议改成：

> **在什么颗粒度上、什么条件下，解卷积输出足以支持细胞状态空间与轨迹推断？**

理由见 §10.8：我们已在另一个接口上观测到强烈预兆——聚合层一致、细胞层不一致。

### 10.3 上高位刊的门票：外部锚定的真值

**致命问题**：第三、四层（轨迹、药物排序）**没有真值**。真实细胞级数据的"真实轨迹"本身是 RNA velocity / Dynamo 的输出，是**另一个模型的预测**。按原提法只能测两个预测的 agreement，不能测 accuracy；若两条分支共用同一套动力学假设，一致率可以很高而结论全错。审稿人必问。

**解法有先例**：Saelens et al., *Nat Biotechnol* 2019（dynverse，轨迹推断方法比较）能上高位刊，靠的是把真值锚在**外部已知生物学**（文献/专家整理的已知分化顺序与拓扑），而不是方法之间的相互比较。

**我们手上恰好有两个现成的外部锚——这是真正的差异化点**：

1. **CNV 克隆谱系**：CNV 是测量量而非模型输出；共享 CNV 事件给出祖先 / 衍生的**方向性**，且与表达轨迹方法完全正交。
   - 表述要收敛：**克隆性**转移可锚定；可逆的可塑性转移（EMT、耐药耐受）不能。别把话说满。
   - **硬前置**：真值的可靠性必须高于被测对象。自算 CNV 在 GSE148071 的 AUROC 仅 **0.7569**，对做下游标签勉强够，**对做 benchmark 真值不够**。这个数字必须先砸实，否则整个 benchmark 是空的。
2. **多部位同患者采样**：GSE131907 的 adjacent normal → primary tumor → lymph node → pleural effusion / brain met，配对关系已用 GEO 官方 SOFT 钉死（10/10，见 §8）。转移播散本身就是恶性进展的**时空顺序锚**。部位梯度结果（癌旁 5% → 原发 19% → LN 46% → 胸水 56%）已证明这个锚能用。

### 10.4 实验设计的缺陷（写 proposal 前必须补）

1. **自参照信息泄漏**：为做到"相同组织、相同细胞"，参考签名天然来自同一批 Xenium/CosMx 数据 → 解卷积会好得失真（签名里已有答案）。必须设计"跨样本参考" vs "自参照"两种条件，否则结论不可信。**笔记完全没提。**
2. **方案 A / B 必须明确选 A**：方案 B（生成伪细胞表达）引入第二重误差，且与解卷积误差**纠缠、无法归因**——这摧毁了 oracle 设计的全部意义（该设计的初衷就是分离误差来源）。选 A（spot-level compositional state space）后问题变窄但更硬。
3. **合成 spot 过于理想 → 结论是乐观上界**：只做简单加和，未模拟捕获效率空间变异、mRNA 扩散、切片截断、3′ 偏好。**便宜的补救**：用已有的 15 张 Visium 切片校验合成 spot 的**组成统计量**是否与真实 Visium 匹配，并在论文里把"上界"写清楚。
4. **scGPT-spatial 分叉未审视**：空间原生预训练模型可能让这个接口问题部分消解。笔记识别到后立即转回原路线，未评估成本 / 收益。
5. **笔记 L773–785 的"药物 A 0.82 → 0.61"是示意性假想数字**，未标注。写 proposal 时直接引用会构成事实错误。
6. **docking / MD 立场前后不一致**：前文严厉批判（对接分数不是结合自由能、不能证明状态反转），后文又列为 MVP 延伸步骤。若保留，只应用于**候选靶点的成药性 / 结构可行性**筛选，与"状态反转证据"严格分开报告。

**一个纠正（相对笔记）**：核心实验的**第二、三层不需要真实空间坐标**。真细胞 → 按生态位混成伪 spot → 解卷积 → 组成 → 状态空间 → 轨迹，用 scRNA-seq 就能跑通。真实单细胞分辨率空间数据（Xenium / CosMx / MERFISH）只在验证第一层（空间富集）时才必需。**所以现有数据够做核心实验。**

### 10.5 竞争风险（2026-09-02 已核实）

> 方法：NCBI E-utilities（esearch / esummary / efetch）+ Crossref API 逐条核对，**不依赖任何 AI 检索**。核验明细见 §10.9。

**结论：gap 保住了，但只剩第三、四层；第一、二层已被占。**

**(1) Li B, …, Qu K. *Nat Methods* 2022;19(6):662-670（PMID 35577954）** —— 标题 "Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution"。**16 种方法 / 45 配对数据集 + 32 模拟数据集**，评两个任务：

- transcript distribution prediction → Tangram / gimVI / SpaGE 最优；
- cell type deconvolution → Cell2location / SpatialDWLS / RCTD 最优。

**它吃掉了第一层和第二层**——但评价指标是"预测表达 vs 实测表达"的相关性，即**重建的准确性**，**完全不涉及下游状态空间、轨迹或药物推断**。
→ 论文 related work 必须显式划界："Li 2022 评的是重建准确性，我们评的是下游可用性"，这是两件事。**第一、二层不能再声称是空白。**

**(2) st2traj（*Bioinformatics*，2026-08-27，PMID 42658029）** —— "Deconvolution-informed trajectory inference for multi-timepoint spatial transcriptomics"。**这正是我们要质疑的那个跨层级接口，已经有人在做方法了**（用 spot-state composition 驱动轨迹推断）。

但它的验证方式是：

> "produced smoother trajectory fields and **stronger agreement with expression-derived marker programs**"

即**正是 §10.3 警告的 agreement-not-accuracy 循环**——没有外部锚。它的 pseudo-spot benchmark 只用来在 5 种解卷积方法里挑一个（DECODE 最优），即只评第一、二层。
→ **对我们是净利好**：证明这个接口正在被使用、却无人用外部锚验证过。这是比"AI 说没人做过"强得多的立项必要性证据，**应在 Introduction 主动引用**。它是 Bioinformatics 的方法论文，高位刊的验证研究位子仍然空着。

**(3) 两轮定向检索的阴性结果**（2023–2026）：

- `benchmark` + `cell state` + `spatial`（标题检索）→ **0 篇**；
- `deconvolution` + `reliability/uncertainty` + `downstream` → 7 篇，**逐条看过无直接竞品**（分别是序列→表达预测、细胞类型注释、bulk DNA 系统发生、PitNET 应用、spot 水平 DE 分析）。

**为什么之前不能委托 AI 核**：笔记里"目前没有找到一篇论文完成完整闭环"这个立项依据来自同一次 AI 检索；而同一份笔记的第一部分已证明该 AI 给出的 4 篇论文错 3 篇。核验结果印证了这个担心——见 §10.9 里 SpatialScope 那条**虚构 DOI**。

### 10.6 发表风险（坦率版）

1. **身份已经变了**：从"药物发现"变成"可靠性评估 / benchmark"。benchmark 在高位刊要的是**发现系统性失效模式 + 给出可操作规范**，不是一张排名表；纯 benchmark 容易被质疑工程增量。
2. **MVP 与 benchmark 的广度要求矛盾**：笔记的 MVP 只锁 1 数据集 + 1 下游模型，而 benchmark 需要 方法 × 数据 × 指标 的覆盖。
3. **阴性结果风险**：笔记预计的结局之一是"不同解卷积方法导致完全不同的状态空间和药物排名"= 可靠性警告。这类结论在生信刊的接受度通常低于新工具论文。
4. **工作量**：这是 1–2 年的主线项目，不是顺手加的一节。

### 10.7 决策顺序（几天的量级，但决定后面一两年）

1. **判定 CNV 能否当真值（最高优先，不依赖 Galaxy）**。§10.5(2) 之后这已经不是"加分项"而是**前提**：agreement 路线被 st2traj 占了，外部锚是我们唯一剩下的可发表性来源。见 §10.3。
2. 研究问题锁死在第三层（+ 第四层）：第一、二层已被 Li 2022 占（§10.5(1)）。
3. 决定是否把 st2traj 作为正面论据写进 Introduction。
4. 再投入工程。

（~~核实竞争文献~~ 已于 2026-09-02 完成，见 §10.5 / §10.9。）

### 10.8 一个已经观察到的预演

`novelty × CNV` 双通道一致性呈**层级结构**（`results/novelty_vs_cnv.json`）：细胞级硬判定 kappa≈0（148071 甚至 −0.01），但 **GSE131907 供体 / 样本级 Spearman 0.6225（p=2.3e-5）**，且部位梯度两法同序。

这个形状——**聚合层一致、细胞层不一致**——几乎必然是那个 benchmark 会跑出的形状。它同时是两件事：支持"颗粒度"提法的证据，以及一个警告——**细胞级对应关系不可主张**（这一点必须写进任何报告）。

### 10.9 引用核验结果（2026-09-02）

来源：NCBI E-utilities（esearch / esummary / efetch）+ Crossref API。**逐条核对，不采信任何 AI 检索结果。**

| 笔记中的引用 | 核验结果 | PMID |
|---|---|---|
| OSCC TC–LE（肿瘤核心/边缘） | ✅ Nat Commun 2023;14:5029，标题期刊全对 | 37596273 |
| SpaRx | ✅ Brief Bioinform 2023;24(5) | 37798249 |
| scMGCA | ✅ **Nat Commun** 2023;14:400（笔记已纠正过的期刊错误，纠正正确） | 36697410 |
| Cell2location | ✅ Nat Biotechnol 2022;40:661-671 | 35027729 |
| Tangram | ✅ Nat Methods 2021;18:1352-1362 | 34711971 |
| scGPT | ✅ Nat Methods 2024;21:1470-1480 | 38409223 |
| CellOracle | ✅ Nature 2023;614:742-751 | 36755098 |
| 解卷积 benchmark | ✅ Nat Commun 2023;14:1548，**"18 方法 / 50 数据集"属实** | 36941264 |
| Saelens 轨迹 benchmark | ✅ Nat Biotechnol 2019;37:547-554 | 30936559 |
| 黑色素瘤 CAF + docking | ✅ 真实，但期刊是 **Comput Biol Med** 2023;167:107597（档次远低于笔记暗示） | 37875042 |
| Li 2022 整合 benchmark | ✅ Nat Methods 2022;19(6):662-670（**第一作者 Li B、通讯 Qu K**，不是"Xu"） | 35577954 |
| **SpatialScope** | ❌ **笔记给的 DOI `10.1038/s41592-023-01937-1` 不存在**。真实论文在 **Nat Commun**，DOI `10.1038/s41467-023-43629-w`，标题 "Integrating spatial and single-cell transcriptomics data using deep generative models with SpatialScope" | — |

**记 11 对、1 条伪 DOI。** 笔记后半部分（战略讨论）的引用明显比前半部分（AI 给的文献清单）可靠——前半部分已知 4 篇错 3 篇。**但 SpatialScope 这条说明后半部分也不能盲信。**

**→ 硬规则：任何进入 proposal / 论文正文的引用，必须走一遍 PMID 或 Crossref 核验。委托 AI 检索的结果一律视为待核。**
