#!/usr/bin/env bash
# /tmp 是易失的。本脚本把上一轮(/tmp)的产物重新链回来，使 25/30/35/37/45/63 可复现。
# 归档在 data/inputs_previous_round/（硬链接，不占额外空间）。
set -u
ARCH=/home/ubuntu/luad_scmg_v2/data/inputs_previous_round
for f in luad_316k_scmg_ready.h5ad luad_316k_scmg_embeddings.npy luad_316k_scmg_embeddings_finetuned.npy \
         luad_316k_scmg_umap_finetuned.csv emb_stage_mal.npy stage_umap.csv diffusion_transition_100.npy; do
  if [ -f "$ARCH/$f" ] && [ ! -e "/tmp/$f" ]; then ln -s "$ARCH/$f" "/tmp/$f"; echo "linked /tmp/$f"; fi
done
for d in scmg_luad_finetuned_opt scmg_stage_ft; do
  if [ -d "$ARCH/$d" ] && [ ! -e "/tmp/$d" ]; then ln -s "$ARCH/$d" "/tmp/$d"; echo "linked /tmp/$d"; fi
done
