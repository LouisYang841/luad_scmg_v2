#!/usr/bin/env bash
# 全流程复现。分两段：
#   PHASE 0  若机器重启过、/tmp 被清空，先恢复上一轮产物的软链接
#   其余按编号顺序执行；49_report_en.py 必须最后跑（它整篇重写 results/REPORT.md）
#
# 依赖的外部数据（全部只读）：
#   /home/ubuntu/luad_invasion/results/...   上游图谱注释、空转丰度表、ground-truth manifest
#   /home/ubuntu/scmg_workspace/SCMG         SCMG 源码
#   /home/ubuntu/scmg_workspace/models/embedder  官方 zero-shot 权重
set -euo pipefail
cd "$(dirname "$0")"
P=/home/ubuntu/qwen-env/bin/python

bash scripts/relink_tmp.sh                                   # PHASE 0：/tmp 易失，先恢复

$P scripts/00_build_labels.py                                # 干净注释表 + 标签审计
$P scripts/10_prep_gpu_matrix.py                             # 316,689x18,108 标准矩阵落盘 (128 s)
$P scripts/20_finetune_unsup.py --epochs 15 --batch 4096     # 无监督适配 (~23 min / L4)
$P scripts/25_audit_prev_work.py                             # 复现上一轮的问题（低显存，可与训练并发）
$P scripts/30_eval_harness.py --mode sweep                   # epoch 选型（label-free 指标）
$P scripts/30_eval_harness.py --mode final --specs \
  "ZS=/home/ubuntu/scmg_workspace/models/embedder/model.pt;FT-luad(leaky)=/tmp/scmg_luad_finetuned_opt/model.pt;FT-stage(leaky)=/tmp/scmg_stage_ft/stage_model.pt;FT-unsup(ep1)=results/finetune_unsup/state_epoch1.pth;FT-unsup(ep15)=results/finetune_unsup/state_epoch15.pth"
$P scripts/37_marker_benchmark.py                            # 与注释半独立的留出供体基准
$P scripts/35_paired_malignancy_axis.py                      # 可识别性 + 同供体配对轴 + 供体级 DE
$P scripts/45_summary_figure.py                              # -> results/summary_rework.png

# --- 模型自带映射（不使用传统聚类标签）---
test -f ref/data/ref_global_cell_state_manifold.h5ad || \
  $P -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='xingjiepan/SCMG_data', repo_type='dataset', allow_patterns=['data/ref_global_cell_state_manifold.h5ad'], local_dir='ref')"
$P scripts/60_model_native_annot.py                          # 全局参考软迁移 + 到参考距离
$P scripts/61_epithelium_novelty.py                          # 类型限制的上皮异常度 + 同供体配对
$P scripts/62_epithelium_calls.py                            # 零分布阈值 -> abn95 判定

# --- 空转生态位 ---
$P scripts/63_model_native_niches.py                         # 交叉表重映射 + 原型匹配 + 消融
$P scripts/65_niche_figure.py                                # -> results/niches/niche_comparison.png

$P scripts/40_report.py                                      # -> results/REPORT_zh.md（中文，1-7 节）
$P scripts/49_report_en.py                                   # -> results/REPORT.md（权威，1-9 节，最后跑）
echo "OK -> results/REPORT.md"
