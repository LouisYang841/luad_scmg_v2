#!/usr/bin/env python3
"""20_finetune_unsup.py — 诚实版 LUAD 领域自适应微调。

与上一轮 (/tmp/finetune_opt.py, /tmp/finetune_stage_gpu.py) 的差别：
  * 损失函数完全照搬官方 calc_loss：contrastive(1000) + l2(0.1) + **recon(1e-3)**
    —— 上一轮把 recon 直接置 0，丢掉了把 latent 钉在基因表达上的那一半。
  * 负样本 = **同 dataset 内随机**（官方规则），不再用任何标签去构造 hard negative。
  * 正样本 = 当前空间的 kNN，图每 --refresh 个 epoch 重建一次（上一轮建一次就不动）。
  * 给 3 个 GSE 新建 dataset token，只让新 token 行 + decoder 可训练，
    预训练的 37 行 token 冻结 -> 保住与全局流形的可比性。
  * 训练用全 316,689 细胞（上一轮只用 67k / 53k 子集）。
  * 留出若干 **完整供体** 做验证，checkpoint 用 label-free 的验证损失挑选。
  * 记录坍缩护栏：effective rank、平均范数、与 zero-shot 空间的 kNN 图重叠。
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch", type=int, default=4096)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--kpos", type=int, default=8)
p.add_argument("--refresh", type=int, default=3)
p.add_argument("--nval-patients", type=int, default=10)
p.add_argument("--outdir", default=f"{L.RES}/finetune_unsup")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--smoke", action="store_true", help="只跑 2 epoch / 少量细胞，用于验证代码")
a = p.parse_args()

os.makedirs(a.outdir, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = L.DEVICE

order = L.load_order()
N_ALL = len(order)
datasets = sorted(order["dataset"].unique())
print("datasets:", datasets, "cells:", N_ALL, flush=True)

# ---------- 供体级 train / val 划分 ----------
rng = np.random.default_rng(a.seed)
val_patients = []
per_ds_cells = []
cnt = order.groupby("dataset")["patient_id"].value_counts()
for ds in datasets:                      # 每个 dataset 里挑细胞数居中的供体做留出
    cand = cnt.loc[ds]
    cand = cand[(cand >= 800) & (cand <= 12000)]
    pick = list(cand.index[:a.nval_patients // len(datasets) + 2])
    val_patients += [f"{ds}|{x}" if not str(x).startswith(ds) else str(x) for x in pick]
val_patients = sorted(set(order.loc[order["patient_id"].isin([v.split("|")[-1] for v in val_patients]), "patient_id"]))
is_val = order["patient_id"].isin(val_patients).values
train_idx = np.where(~is_val)[0]; val_idx = np.where(is_val)[0]
if a.smoke:
    train_idx = train_idx[:40000]; val_idx = val_idx[:6000]
print(f"val patients={len(val_patients)} val_cells={len(val_idx)} train_cells={len(train_idx)}", flush=True)

# ---------- 显存里的标准矩阵 ----------
X_all = L.x_to_gpu()
if a.smoke:
    X_all = X_all[:40000]
    train_idx = train_idx[train_idx < 40000]; val_idx = val_idx[val_idx < 40000]
Xtr_gpu = torch.as_tensor(train_idx, dtype=torch.long, device=dev)
Xva_gpu = torch.as_tensor(val_idx, dtype=torch.long, device=dev)

# ---------- 模型 + 新 dataset token ----------
model = L.load_embedder(f"{L.BASE_MODEL}/model.pt")
sd = torch.load(f"{L.BASE_MODEL}/best_state_dict.pth", map_location=dev, weights_only=True)
model.load_state_dict(sd)
model, ids, trainable_rows = L.add_dataset_tokens(model, datasets)
model = L.lock_pretrained_rows(model, n_pretrained=37)
dsname_to_id = torch.as_tensor([ids[d] for d in datasets], dtype=torch.long, device=dev)
ds_local = pd.Categorical(order["dataset"].values).codes            # 0..2 per cell
ds_local_tr = ds_local[train_idx]
# 每个 dataset 的 train 细胞查找表（负样本 = 同 dataset 随机，官方规则）
lookup, starts, counts = [], [], []
for j, d in enumerate(datasets):
    ii = np.where(ds_local_tr == j)[0]
    starts.append(len(lookup)); counts.append(len(ii)); lookup.append(ii)
lookup = torch.as_tensor(np.concatenate(lookup), dtype=torch.long, device=dev)
starts = torch.as_tensor(starts, dtype=torch.long, device=dev)
counts = torch.as_tensor(counts, dtype=torch.long, device=dev)
ds_of_train = torch.as_tensor(ds_local_tr, dtype=torch.long, device=dev)

token_of_train = dsname_to_id[ds_of_train]        # 每个 train 细胞的 dataset token

# ---------- 优化器（官方 AdamW 默认 betas/wd，只把 lr 降到微调量级） ----------
opt = torch.optim.AdamW([
    {"params": list(model.encoder.parameters()), "lr": a.lr},
    {"params": list(model.decoder.parameters()), "lr": a.lr},
    {"params": [model.dataset_emb], "lr": 1e-3}],   # 只有新增 3 行会真正收到梯度
                        weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda", enabled=True)


@torch.no_grad()
def embed(idx_gpu, batch=8192):
    model.eval()
    out = []
    for s in range(0, len(idx_gpu), batch):
        Xb = X_all[idx_gpu[s:s + batch]].float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(Xb).float().cpu())
    model.train()
    return torch.cat(out).numpy().astype(np.float32)


def build_edges(idx_cpu):
    """当前空间 kNN 图 -> (anchor, positive) 边（下标是 idx_cpu 内的局部下标）。"""
    ig = torch.as_tensor(idx_cpu, dtype=torch.long, device=dev)
    E = embed(ig)
    D, I = L.knn_on_gpu(torch.as_tensor(E, device=dev), k=a.kpos + 1)
    I = I[:, 1:]
    anc = np.repeat(np.arange(len(idx_cpu)), a.kpos)
    return np.stack([anc, I.reshape(-1)], 1).astype(np.int32), E


# ---------- 固定的 label-free 验证损失（验证细胞 + 由 zero-shot 图定义的邻居） ----------
ZS = np.load("/tmp/luad_316k_scmg_embeddings.npy")
zs_val = ZS[val_idx]
_, I = L.knn_on_gpu(torch.as_tensor(zs_val, device=dev), k=a.kpos + 1)
val_edges = np.stack([np.repeat(np.arange(len(val_idx)), a.kpos),
                      I[:, 1:].reshape(-1)], 1).astype(np.int32)
del I
va_ds_local = pd.Categorical(order["dataset"].values[val_idx]).codes
va_lookup = [torch.as_tensor(np.where(va_ds_local == j)[0], dtype=torch.long, device=dev)
             for j in range(len(datasets))]
va_counts = torch.as_tensor([len(x) for x in va_lookup], dtype=torch.long, device=dev)
va_starts = torch.as_tensor([0] + np.cumsum([len(x) for x in va_lookup[:-1]]).tolist(), dtype=torch.long, device=dev)
va_lookup_t = torch.as_tensor(np.concatenate([x.cpu().numpy() for x in va_lookup]), dtype=torch.long, device=dev)
va_ds_t = torch.as_tensor(va_ds_local, dtype=torch.long, device=dev)
va_token = dsname_to_id[va_ds_t]


# ---------- 训练 ----------
hist = []
edges, E0 = build_edges(train_idx)
print(f"edges {edges.shape}", flush=True)
best = float("inf")
t_start = time.time()
for ep in range(a.epochs):
    if ep > 0 and ep % a.refresh == 0:
        edges, _ = build_edges(train_idx)
        print(f"  graph refreshed at epoch {ep+1} (edges {edges.shape})", flush=True)
    model.train()
    ep_loss = {"c": [], "l2": [], "r": []}
    n = len(edges) // a.batch
    pbar = torch.randperm(len(edges), device=dev)[:n * a.batch]
    e_all = torch.as_tensor(edges, dtype=torch.long, device=dev)[pbar]
    t0 = time.time()
    for s in range(n):
        e = e_all[s * a.batch:(s + 1) * a.batch]
        ia, ip = e[:, 0], e[:, 1]
        ds_a = ds_of_train[ia]
        rn = (torch.rand(a.batch, device=dev) * counts[ds_a]).long()
        ine = lookup[starts[ds_a] + rn]
        Xa = X_all[Xtr_gpu[ia]].float(); Xp = X_all[Xtr_gpu[ip]].float(); Xn = X_all[Xtr_gpu[ine]].float()
        Xp_clean = Xp
        Xa, Xp, Xn = L.official_dropout(Xa, Xp, Xn)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            Za, Zp, Zn = model(Xa), model(Xp), model(Xn)
            l2 = (L.l2_loss(Za) + L.l2_loss(Zp) + L.l2_loss(Zn)) * 0.1
            con = (L.contrastive_loss(Za, Zp, Zn) + L.contrastive_loss(Zp, Za, Zn)) * 1000
            rec = L.recon_loss(Xp_clean.float(), L.decode_fast(model, Zp, token_of_train[ip])) * 1e-3
            loss = con + l2 + rec
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if model.dataset_emb.grad is not None:            # 预训练 token 行保持冻结（必须在 step 之前）
            model.dataset_emb.grad[:model._n_pretrained_rows] = 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update()
        ep_loss["c"].append(con.item()); ep_loss["l2"].append(l2.item()); ep_loss["r"].append(rec.item())
        if s % 50 == 0:
            print(f"  ep{ep+1} {s}/{n} c={con.item():.1f} l2={l2.item():.2f} rec={rec.item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # ---------------- 验证（label-free，留出供体） ----------------
    Ev = embed(Xva_gpu)
    model.eval()
    vc, vl, vr = [], [], []
    with torch.no_grad():
        for s in range(len(val_edges) // a.batch):
            e = torch.as_tensor(val_edges[s * a.batch:(s + 1) * a.batch], dtype=torch.long, device=dev)
            ia, ip = e[:, 0], e[:, 1]
            ds_a = va_ds_t[ia]
            rn = (torch.rand(len(ia), device=dev) * va_counts[ds_a]).long()
            ine = va_lookup_t[va_starts[ds_a] + rn]
            Xa = X_all[Xva_gpu[ia]].float(); Xp = X_all[Xva_gpu[ip]].float(); Xn = X_all[Xva_gpu[ine]].float()
            Za, Zp, Zn = model(Xa), model(Xp), model(Xn)
            vc.append((L.contrastive_loss(Za, Zp, Zn) + L.contrastive_loss(Zp, Za, Zn)).item() * 1000)
            vl.append((L.l2_loss(Za) + L.l2_loss(Zp) + L.l2_loss(Zn)).item() * 0.1)
            vr.append(L.recon_loss(Xp.float(), L.decode_fast(model, Zp, va_token[ip])).item() * 1e-3)
    # recon 的相关系数（在留出细胞的真实 X 上）
    with torch.no_grad():
        idxs = torch.as_tensor(np.arange(min(8000, len(val_idx))), dtype=torch.long, device=dev)
        Xv = X_all[Xva_gpu[idxs]].float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            Zv = model(Xv); Pv = L.decode_fast(model, Zv, dsname_to_id[va_ds_t[idxs]])
        a_, b_ = Xv.cpu().numpy().ravel(), Pv.float().cpu().numpy().ravel()
        rec_corr = float(np.corrcoef(a_, b_)[0, 1])
    # 与 zero-shot 空间的 kNN 图重叠（流形保持度，不参与选型，只记录）
    overlap = float("nan")
    if len(val_idx) <= 30000:
        overlap = L.knn_graph_overlap(zs_val, Ev, None, k=20)
    vloss = float(np.mean(vc) + np.mean(vl) + np.mean(vr))
    er = L.effective_rank(Ev)
    row = dict(epoch=ep + 1, train_c=float(np.mean(ep_loss["c"])), train_l2=float(np.mean(ep_loss["l2"])),
               train_rec=float(np.mean(ep_loss["r"])), val_c=float(np.mean(vc)), val_rec=float(np.mean(vr)),
               val_recon_corr=rec_corr, val_loss=vloss, eff_rank_val=er,
               mean_norm_val=float(np.linalg.norm(Ev, axis=1).mean()), knn_overlap_vs_zs=overlap,
               secs=time.time() - t0)
    hist.append(row); print("== " + json.dumps(row), flush=True)
    with open(f"{a.outdir}/history.json", "w") as f:
        json.dump(hist, f, indent=2)
    torch.save(model.state_dict(), f"{a.outdir}/state_epoch{ep+1}.pth")
    if vloss < best:
        best = vloss
        torch.save(model.state_dict(), f"{a.outdir}/best_state_dict.pth")
        torch.save(model, f"{a.outdir}/best_model.pt")
        json.dump({"dataset_id_map": model.dataset_id_map, "best_epoch": ep + 1, "best_val_loss": best, "val_patients": val_patients,
                   "config": vars(a)}, open(f"{a.outdir}/best_meta.json", "w"), indent=2)
        print("   ^ new best (val_loss %.3f)" % best, flush=True)

print(f"total {time.time()-t_start:.0f}s  best val_loss {best:.3f}")
# 最终留出细胞的 embedding（供 30_eval 使用）
np.save(f"{a.outdir}/emb_val_last.npy", embed(Xva_gpu))
