#!/usr/bin/env python3
"""luadlib.py — LUAD/SCMG 返工共用的工具函数。

设计原则（针对上一轮的三个方法问题）：
  1. 任何"标签"（stage / 区室）都只用于**事后评估**，绝不进入训练目标；
  2. 推断单元 = patient，所有泛化评估都走 leave-one-patient-out；
  3. 与冻结的全局 zero-shot 空间保持可比（CKA / kNN 图重叠 作为 label-free 模型选择指标）。
"""
import os, sys, json, time
sys.path.insert(0, "/home/ubuntu/scmg_workspace/SCMG")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import torch.serialization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from scipy.stats import spearmanr

ROOT = "/home/ubuntu/luad_scmg_v2"
DAT = f"{ROOT}/data"
RES = f"{ROOT}/results"
BASE_MODEL = "/home/ubuntu/scmg_workspace/models/embedder"
STD_GENES = np.load(f"{DAT}/std_genes.npy") if os.path.exists(f"{DAT}/std_genes.npy") else None
NG = 18108
DEVICE = "cuda"

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ------------------------------------------------------------------ data ----
def load_X_memmap():
    return np.memmap(f"{DAT}/X_std_18108.f16", dtype=np.float16, mode="r",
                     shape=(len(load_order()), NG))


def load_order():
    return pd.read_csv(f"{DAT}/cell_order.csv")


def x_to_gpu(n_cells=None, verbose=True):
    """整块标准矩阵搬进显存 (float16, ~11.5GB)。"""
    t0 = time.time()
    X = torch.from_numpy(np.asarray(load_X_memmap())).to(DEVICE)
    if verbose:
        print(f"X_all {tuple(X.shape)} {X.dtype} -> GPU in {time.time()-t0:.0f}s "
              f"({torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)
    return X


# ----------------------------------------------------------------- model ----
def load_embedder(path, device=DEVICE):
    from scmg.model.contrastive_embedding import CellEmbedder
    torch.serialization.add_safe_globals([CellEmbedder])
    m = torch.load(path, map_location=device, weights_only=False)
    m.to(device)
    return m


def load_any(path, extra_token_names=(), device=DEVICE):
    """统一加载：全模型 .pt 或 state_dict .pth（以官方模型为骨架）。

    state_dict 不含 dataset_id_map，因此微调过的 ckpt（行数变多）需要按确定顺序
    （= add_dataset_tokens 的追加顺序）把新增 token 名字补回去。
    """
    if path.endswith(".pt"):
        return load_embedder(path, device)
    m = load_embedder(f"{BASE_MODEL}/model.pt", device)
    sd = torch.load(path, map_location=device, weights_only=True)
    n = sd["dataset_emb"].shape[0]
    k = m.dataset_emb.data.shape[0]
    if n > k:
        m.dataset_emb.data = torch.cat(
            [m.dataset_emb.data, torch.zeros(n - k, m.dataset_emb.data.shape[1], device=m.dataset_emb.device)], 0)
        for i, nm in enumerate(list(extra_token_names)[:n - k]):
            m.dataset_id_map[nm] = k + i
    m.load_state_dict(sd)
    return m


def add_dataset_tokens(model, new_names, init_from=("Sikkema_Lung_HS_2023:core",
                                                    "He_LungDev_HS_2022:all",
                                                    "Eraslan_MultiTissue_HS_2022:all"),
                       seed=0):
    """给 LUAD 新增 dataset token（decoder 侧条件），返回 (model, name->id, 可训练行索引)。

    新 token 用已预训练的肺/多组织 token 的均值 + 小噪声初始化，使初始重建不至于崩掉；
    预训练的 37 行保持冻结（梯度置零），只让新增行可训练 -> 保住全局流形语义。
    """
    emb = model.dataset_emb
    dev = emb.device
    K = emb.data.shape[0]
    ids = dict(model.dataset_id_map)
    base_rows = [ids[n] for n in init_from if n in ids]
    g = torch.Generator(device="cpu").manual_seed(seed)
    mean = emb.data[base_rows].mean(0)
    noise = 0.01 * torch.randn(len(new_names), emb.data.shape[1], generator=g).to(dev)
    new = (mean.unsqueeze(0).to(dev) + noise)
    emb.data = torch.cat([emb.data, new], 0)
    for i, n in enumerate(new_names):
        ids[n] = K + i
    model.dataset_id_map = ids
    trainable_rows = list(range(K, K + len(new_names)))
    return model, ids, trainable_rows


def lock_pretrained_rows(model, n_pretrained=37):
    """记录冻结行数；真正的梯度置零由训练循环在 scaler.step 之前执行。"""
    model._n_pretrained_rows = n_pretrained
    model.dataset_emb.requires_grad_(True)
    return model


def decode_fast(model, Z, ds_ids):
    """等价于 CellEmbedder.decode，但用预先算好的整数 token（避免每步 Python 循环）。"""
    d = torch.cat([Z, model.dataset_emb[ds_ids]], dim=1)
    return F.softplus(model.decoder(d))


# ----------------------------------------------------------- losses (官方) --
def l2_loss(Z):
    return torch.mean(torch.sum(Z.pow(2), dim=1))


def contrastive_loss(a, p, n):
    dp = torch.sqrt(torch.sum((a - p).pow(2), dim=1) + 1e-8)
    dn = torch.sqrt(torch.sum((a - n).pow(2), dim=1) + 1e-8)
    return torch.mean(F.softplus(dp - dn))


def recon_loss(X_exp, X_pred):
    return torch.sum(torch.square(X_pred - X_exp)) / X_exp.shape[0]


def official_dropout(Xa, Xp, Xn, max_rate=0.3):
    """复刻官方 edge_batch_to_Xs：以 30% 概率施加 rate~U(0,0.3) 的输入 dropout。"""
    r = np.random.uniform(0, 1)
    if r < max_rate:
        rate = r
        return (Xa * (torch.rand_like(Xa) > rate),
                Xp * (torch.rand_like(Xp) > rate),
                Xn * (torch.rand_like(Xn) > rate))
    return Xa, Xp, Xn


# --------------------------------------------------------------- metrics ----
def cka(A, B, subsample=6000, seed=0):
    """linear CKA (逐维中心化后)。"""
    rng = np.random.default_rng(seed)
    if len(A) > subsample:
        i = rng.choice(len(A), subsample, replace=False)
        A, B = A[i], B[i]
    A = A - A.mean(0); B = B - B.mean(0)
    A = A.astype(np.float32); B = B.astype(np.float32)
    return float(np.linalg.norm(A.T @ B) ** 2 /
                 (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def knn_on_gpu(X, k=30, query=None):
    """faiss-GPU 精确 kNN（含自身）。X: (N,d) float32 GPU tensor."""
    import faiss
    res = faiss.StandardGpuResources()
    idx = faiss.IndexFlatL2(X.shape[1])
    g = faiss.index_cpu_to_gpu(res, 0, idx)
    Xc = np.ascontiguousarray(X.detach().cpu().numpy().astype(np.float32))
    g.add(Xc)
    Q = Xc if query is None else query
    D, I = g.search(np.ascontiguousarray(Q.astype(np.float32)), k)
    return D, I


def knn_graph_overlap(A, B, idx, k=20, cap=12000, seed=0):
    """同一批细胞在两个空间里 kNN 集合的平均重叠（label-free 的流形保持度）。"""
    rng = np.random.default_rng(seed)
    if len(A) > cap:
        i = rng.choice(len(A), cap, replace=False)
        A, B = np.asarray(A)[i], np.asarray(B)[i]
    A = np.asarray(A, dtype=np.float32); B = np.asarray(B, dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    nA = NearestNeighbors(n_neighbors=k + 1).fit(A)
    nB = NearestNeighbors(n_neighbors=k + 1).fit(B)
    _, IA = nA.kneighbors(A); _, IB = nB.kneighbors(B)
    IA, IB = IA[:, 1:], IB[:, 1:]
    return float(np.mean([len(set(IA[i]) & set(IB[i])) / k for i in range(len(IA))]))


def lopo_probe(X, y, groups, cap=2500, seed=0, n_pcs=64, test_groups=None, max_test_groups=25,
               return_detail=False):
    """leave-one-patient-out 线性探针。X 行子集由 cap 平衡抽样决定。

    返回 (balanced_accuracy, macro_f1)。这是把 patient 当推断单元的唯一诚实做法。
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(str); groups = np.asarray(groups).astype(str)
    keep = []
    for c in np.unique(y):
        i = np.where(y == c)[0]
        keep.append(rng.choice(i, min(cap, len(i)), replace=False))
    keep = np.concatenate(keep); keep.sort()
    Xs, ys, gs = X[keep], y[keep], groups[keep]
    if test_groups is None:
        test_groups = np.unique(gs)
        if len(test_groups) > max_test_groups:          # 控制运行时间：随机抽供体做留出折
            test_groups = np.random.default_rng(seed + 1).choice(
                test_groups, max_test_groups, replace=False)
    pc = PCA(n_pcs, random_state=0).fit_transform(Xs.astype(np.float32))
    pc = StandardScaler().fit_transform(pc)
    preds = np.empty(len(ys), dtype=object); mask = np.zeros(len(ys), bool)
    for g in test_groups:
        te = gs == g
        tr = ~te
        if te.sum() < 5 or len(set(ys[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1500, C=1.0, class_weight="balanced").fit(pc[tr], ys[tr])
        preds[te] = clf.predict(pc[te]); mask[te] = True
    if mask.sum() == 0:
        return (float("nan"), float("nan"))
    ba = balanced_accuracy_score(ys[mask], preds[mask])
    f1 = f1_score(ys[mask], preds[mask], average="macro")
    if return_detail:
        return ba, f1, int(mask.sum()), len(np.unique(gs[mask]))
    return float(ba), float(f1)


def effective_rank(X, subsample=8000, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    X = X - X.mean(0)
    s = np.linalg.svd(X.astype(np.float32), compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def silhouette_sub(X, y, cap=2000, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        i = np.where(y == c)[0]
        keep.append(rng.choice(i, min(cap, len(i)), replace=False))
    keep = np.concatenate(keep)
    return float(silhouette_score(X[keep].astype(np.float32), np.asarray(y)[keep]))
