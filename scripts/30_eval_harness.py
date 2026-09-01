#!/usr/bin/env python3
"""30_eval_harness.py — 诚实评估台。

对比四个空间：
  ZS          官方全局模型，zero-shot 投影（不做任何适配）
  FT-luad     上一轮 broad-label hard-negative 微调（标签泄漏）
  FT-stage    上一轮 stage-aware hard-negative 微调（标签泄漏 + 只在 53k "Malignant" 上训）
  FT-unsup    本次：官方三项损失、同-dataset 负样本、全 316k、供体级留出、无标签

指标分两类，绝不混用：
  (A) 选型/泛化（label-free，不看任何注释）：masked-gene recovery、recon corr
  (B) 代价（与全局流形的可比性）：与 ZS 的 kNN 图重叠、linear CKA、effective rank
  (C) 事后解释（只在留出供体上做 leave-one-patient-out 探针，绝不 random split）：
      恶性 vs 正常AT2（GSE148071 dataset 内）、细胞类型 6 类、stage 6 类（标注为混淆不可用）

epoch 扫描模式 (--sweep) 只在留出供体上算，用于挑最佳 epoch。
"""
import argparse, json, os, sys, glob, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["sweep", "final"], default="final")
p.add_argument("--ckpt-dir", default=f"{L.RES}/finetune_unsup")
p.add_argument("--leaky-luad", default="/tmp/scmg_luad_finetuned_opt/model.pt")
p.add_argument("--leaky-stage", default="/tmp/scmg_stage_ft/stage_model.pt")
p.add_argument("--limit-cells", type=int, default=60000)
p.add_argument("--tag", default="FT-unsup")
p.add_argument("--ckpt", default=None, help="final 模式下要评估的权重（state_dict .pth）；默认取 best_state_dict.pth")
p.add_argument("--specs", default=None,
               help='final 模式自定义模型表，形式 "tag=path;tag=path"（覆盖默认四个）')
a = p.parse_args()

dev = L.DEVICE
BASEPATH = L.BASE_MODEL
order = L.load_order()
X_all = L.x_to_gpu()

# 留出供体（与训练同一份划分，读 best_meta.json）
meta_path = f"{a.ckpt_dir}/best_meta.json"
val_patients = json.load(open(meta_path))["val_patients"] if os.path.exists(meta_path) else []
is_val = order["patient_id"].isin(val_patients).values
val_pos = torch.as_tensor(np.where(is_val)[0], dtype=torch.long, device=dev)
print(f"held-out donors: {len(val_patients)} patients / {int(is_val.sum())} cells", flush=True)

DS_TOKEN_FALLBACK = "Sikkema_Lung_HS_2023:core"
ds_local = pd.Categorical(order["dataset"].values).codes
ds_names = order["dataset"].values          # 每行的 dataset 名字（token 查表用）
datasets = list(pd.Categorical(order["dataset"].values).categories)


@torch.no_grad()
def embed_subset(model, idx, token_ids, batch=8192):
    out = []
    for s in range(0, len(idx), batch):
        Xb = X_all[idx[s:s + batch]].float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(Xb).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


@torch.no_grad()
def masked_gene_recovery(model, idx, token_ids, frac=0.2, seed=7, n_cells=8000):
    """随机遮住每个细胞 frac 比例的基因 -> 编码 -> 解码 -> 只在被遮住的位点上算恢复质量。
    这是 label-free 的泛化指标（类似语言模型困惑度），可以合法地用来选 checkpoint。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = min(n_cells, len(idx))
    sel = idx[torch.randperm(len(idx), generator=g)[:n].to(dev)]
    X = X_all[sel].float()
    m = (torch.rand(X.shape, generator=g) < frac).to(dev)        # True = 遮住
    Xm = X * (~m).float()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        Z = model(Xm)
        P = L.decode_fast(model, Z, token_ids[sel])
    mm = m & (X > 0)                                             # 只看真实有表达、又被遮住的位点
    if mm.sum() < 1000:
        mm = m
    tgt = X[mm]; pr = P[mm]
    return dict(mask_recovery_corr=float(np.corrcoef(tgt.cpu().numpy(), pr.cpu().numpy())[0, 1]),
                mask_recovery_rmse=float(torch.sqrt(torch.mean((tgt - pr) ** 2)).item()),
                n_masked=int(mm.sum()))


@torch.no_grad()
def plain_recon(model, idx, token_ids, n_cells=6000):
    sel = idx[:min(n_cells, len(idx))]
    X = X_all[sel].float()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        P = L.decode_fast(model, model(X), token_ids[sel])
    x = X.cpu().numpy().ravel(); q = P.float().cpu().numpy().ravel()
    return float(np.corrcoef(x, q)[0, 1])


def probes(emb, mask_cells):
    """全部走 leave-one-patient-out。"""
    o = order.iloc[np.where(mask_cells)[0]]
    y_epi = o["malignant_call"].where(o["malignant_call"] != "non_epithelial", "other")
    out = {}
    # (1) dataset 内的恶性判定：只用 GSE148071（唯一同时含 malignant 与正常AT2 且样本量够的）
    for ds in ["GSE148071", "GSE189357"]:
        sub = (o["dataset"] == ds) & y_epi.isin(["malignant", "normal_at2"])
        if sub.sum() > 500 and o.loc[sub, "malignant_call"].nunique() > 1:
            ba, f1 = L.lopo_probe(emb[sub.values], y_epi[sub.values],
                                  o.loc[sub, "patient_id"].values, cap=1200, max_test_groups=12)
            out[f"LOPO_malignant_vs_AT2__{ds}"] = round(ba, 4)
    # (2) 细胞类型 6 类（免疫/基质），跨供体
    ct = o["celltype_probe"]
    keep = ct.isin(["T cell", "Macrophages", "B cell", "Plasma cell", "Fibroblast", "Endothelial"])
    ba, f1 = L.lopo_probe(emb[keep.values], ct[keep.values], o.loc[keep, "patient_id"].values,
                          cap=1200, max_test_groups=12)
    out["LOPO_celltype6"] = round(ba, 4); out["LOPO_celltype6_f1"] = round(f1, 4)
    # (3) stage 6 类 —— 与 dataset/组织部位完全混淆，仅作为"上一轮优化目标"的复现指标
    st = o["stage"]
    mst = st.isin(["Normal", "AAH", "AIS", "MIA", "IAC", "LNM"])
    ba, f1 = L.lopo_probe(emb[mst.values], st[mst.values], o.loc[mst, "patient_id"].values,
                          cap=900, max_test_groups=12)
    out["LOPO_stage6_CONFOUNDED"] = round(ba, 4)
    # silhouette（random-split 语义，上一轮用的就是它 -> 保留以便对照，但注明循环性）
    out["sil_stage6_leaky"] = round(L.silhouette_sub(emb[mst.values], st[mst.values], cap=1200), 4)
    return out


def load_any(ckpt):
    return L.load_any(ckpt, extra_token_names=datasets)


def run_one(tag, ckpt):
    """返回 (留出供体 embedding, 指标行)。
    重建类指标用两种 token 各算一次：
      *_luad_tok   : 给该模型一个未训练/已训练的 LUAD token（公平：大家都有机会）
      *_lungtok    : 用预训练的肺 token（官方模型本来就能用）
    """
    model = load_any(ckpt)
    model.eval()
    m1, ids1, _ = L.add_dataset_tokens(model, datasets, seed=0)
    tok_luad = torch.as_tensor([ids1[d] for d in ds_names], dtype=torch.long, device=dev)
    emb = embed_subset(model, val_pos, tok_luad)
    row = dict(model=tag, n_val_cells=int(len(val_pos)))
    row["recon_corr"] = round(plain_recon(model, val_pos, tok_luad), 4)
    row.update({f"{k}_luadTok": round(v, 4) for k, v in
                masked_gene_recovery(model, val_pos, tok_luad).items()})
    ids0 = L.load_embedder(f"{L.BASE_MODEL}/model.pt").dataset_id_map
    tok0 = torch.as_tensor([ids0[DS_TOKEN_FALLBACK]] * len(ds_names), dtype=torch.long, device=dev)
    row["recon_corr_lungtok"] = round(plain_recon(model, val_pos, tok0), 4)
    row.update({f"{k}_lungTok": round(v, 4) for k, v in
                masked_gene_recovery(model, val_pos, tok0).items()})
    row["mean_norm"] = round(float(np.linalg.norm(emb, axis=1).mean()), 3)
    del model
    torch.cuda.empty_cache()
    return emb, row


def main():
    rows, embs = [], {}
    if a.mode == "sweep":
        ckpts = [f"{L.BASE_MODEL}/model.pt"] + sorted(
            glob.glob(f"{a.ckpt_dir}/state_epoch*.pth"),
            key=lambda s: int(s.split("epoch")[-1].split(".")[0]))
        for c in ckpts:
            tag = "ZS" if c.endswith("model.pt") and BASEPATH in c else f"epoch{c.split('epoch')[-1].split('.')[0]}"
            e, r = run_one(tag, c)
            rows.append(r); embs[tag] = e
            print(json.dumps(r), flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(f"{L.RES}/sweep_metrics.csv", index=False)
        print(df.to_string(index=False))
        print("\n>>> 按 mask_recovery_corr_lungTok 选出的最佳 epoch:",
              df.sort_values("mask_recovery_corr_lungTok").iloc[-1]["model"],
              " / 按 recon_corr:", df.sort_values("recon_corr").iloc[-1]["model"])
        return

    # ---- final: 四个模型全量对比 ----
    ZS_emb_all = np.load("/tmp/luad_316k_scmg_embeddings.npy")
    if a.specs:
        specs = [(t, p) for t, p in (s.split("=", 1) for s in a.specs.split(";"))]
    else:
        ckpt_final = a.ckpt or f"{a.ckpt_dir}/best_state_dict.pth"
        specs = [("ZS", f"{L.BASE_MODEL}/model.pt"),
                 ("FT-luad(leaky)", a.leaky_luad),
                 ("FT-stage(leaky)", a.leaky_stage),
                 (a.tag, ckpt_final)]
    for tag, ck in specs:
        t0 = time.time()
        e, r = run_one(tag, ck)
        # 事后探针（只在留出供体上）
        r.update(probes(e, is_val))
        # 流形保持度：与 zero-shot 空间比较（同一批留出细胞）
        r["CKA_vs_ZS"] = round(L.cka(e, ZS_emb_all[np.where(is_val)[0]]), 4)
        r["knn20_overlap_vs_ZS"] = round(L.knn_graph_overlap(e, ZS_emb_all[np.where(is_val)[0]], None, 20), 4)
        r["eff_rank"] = round(L.effective_rank(e), 2)
        r["embed_secs"] = round(time.time() - t0, 1)
        rows.append(r); print(json.dumps(r), flush=True)
        np.save(f"{L.RES}/emb_val_{tag}.npy", e)
    df = pd.DataFrame(rows)
    df.to_csv(f"{L.RES}/comparison_models.csv", index=False)
    print("\n" + df.T.to_string())


if __name__ == "__main__":
    main()
