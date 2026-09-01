#!/usr/bin/env bash
# 全流程复现。注意 /tmp 下的上一轮产物若已被清理，25/30 中的 leaky 对比行会自动跳过或报错。
set -euo pipefail
cd "$(dirname "$0")"
P=/home/ubuntu/qwen-env/bin/python
$P scripts/00_build_labels.py
$P scripts/10_prep_gpu_matrix.py
$P scripts/20_finetune_unsup.py --epochs 15 --batch 4096
$P scripts/25_audit_prev_work.py
$P scripts/30_eval_harness.py --mode sweep
$P scripts/37_marker_benchmark.py
$P scripts/35_paired_malignancy_axis.py
$P scripts/40_report.py
$P scripts/45_summary_figure.py
echo "OK -> results/REPORT.md, results/summary_rework.png"
