#!/usr/bin/env python3
"""49_report_en.py -- regenerate results/REPORT.md entirely in English.

Reads the artefacts produced by 00/20/25/30/35/37/60/61/62/63 and emits one
self-contained report. English only on purpose: Chinese text inside generated
scripts kept breaking on shell quoting/escaping.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

RES = L.RES
md = []
w = lambda *a: md.append(" ".join(str(x) for x in a))
J = lambda p: json.load(open(p)) if os.path.exists(p) else {}
C = lambda p, **k: pd.read_csv(p, **k) if os.path.exists(p) else pd.DataFrame()

w("# LUAD x SCMG fine-tuning rework -- report\n")
w("Generated:", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"), "\n")
w("Reproducibility: `run_all.sh` (steps 00 -> 65). Base model = SCMG official global-atlas")
w("embedder (37 dataset tokens, developmental/multi-tissue atlases, NO tumour dataset),")
w("so the previous round's 'lung-cancer model -> LUAD model' framing was incorrect.\n")

# ------------------------------------------------------------- 1. labels --
ann = C(f"{RES}/cell_annotations_clean.csv", index_col=0)
order = C(f"{L.DAT}/cell_order.csv")
w("## 1. Identifiability: which comparisons simply cannot be made\n")
w("### 1a. `stage` is a sample-level attribute, aliased with dataset and tissue site\n")
st = order.groupby("stage").agg(n_cells=("patient_id", "size"), n_donors=("patient_id", "nunique"),
                                n_datasets=("dataset", "nunique"),
                                datasets=("dataset", lambda x: "|".join(sorted(set(x)))),
                                tissue_sites=("tissue_site", lambda x: "|".join(sorted(set(x)))))
w("```"); w(st.to_string()); w("```")
w("\n> AAH = 1 donor, MIA = 2, AIS = 3 in the mapping table used by the previous fine-tune.")
w("> Normal and LNM come only from GSE131907, AAH/AIS/MIA only from GSE189357.")
w("> A `stage` silhouette therefore measures dataset/donor identity, not progression.")
w("> NOTE: a second, better column `stage_true` exists inside Pure_Tumor_subatlas.rds")
w("> (AAH 11 / AIS 11 / MIA 11 / IAC 27 samples) and disagrees with the mapping table,")
w("> which collapsed all of GSE148071 to IAC. The old fine-tune labels were lossy.\n")
w("### 1b. `broad_cell_type == 'Malignant'` is not a malignancy call\n")
w("```"); w(pd.crosstab(ann["broad_cell_type"], ann["compartment_clean"]).to_string()); w("```")
mm = ann["broad_cell_type"] == "Malignant"
w("```"); w(pd.crosstab(ann.loc[mm, "compartment_clean"], ann.loc[mm, "dataset"]).to_string()); w("```")
w("\n> Per-dataset composition of the 'Malignant' bucket. Authoritative cell counts, verified")
w("> identical across `/tmp/luad_316k_scmg_ready.h5ad`.obs, `data/cell_order.csv` and")
w("> `results/cell_annotations_clean.csv` (0 mismatches on 316,689 cells); dataset totals")
w("> GSE131907 183,736 / GSE189357 95,624 / GSE148071 37,329:")
w("> * GSE131907 contributes 23,350 'Malignant' cells of which **0** are annotated malignant")
w(">   epithelium (all 23,350 are Normal Alveolar Epithelial);")
w("> * GSE148071 contributes 21,428, of which 13,412 ARE malignant epithelium and 8,016 are not;")
w("> * GSE189357 contributes 8,876, of which **all** are malignant epithelium.")
w("> So the field's meaning really does shift by dataset, and the two annotation fields")
w("> contradict each other for 31,366 cells. Section 9 shows the GSE131907 cells are in fact")
w("> abnormal, i.e. there the *fine* label is wrong rather than the broad one.")
w("> Also: `Tumor_Tumor C0..C13` are literally renamed Seurat clusters")
w("> (build_11_clones.R:21 `paste0('Tumor C', seurat_clusters)`) -- clones in name only.\n")

# ----------------------------------------------------------- 2. old audit --
A = J(f"{RES}/audit_prev_work.json")
if A:
    w("## 2. Reproducible audit of the previous round (`scripts/25_audit_prev_work.py`)\n")
    w("### 2a. Circular evaluation: hold out donors and the gain disappears\n")
    w("```"); w(pd.DataFrame(A["1_stage_probe_leakage"]).to_string(index=False)); w("```")
    w(f"\n> chance = {A['1_chance']}, {A['1_held_out_donors']} donors fully held out.")
    w("> `random_split_sample_leak` is the equivalent of what was reported before")
    w("> (same donor on both sides of the split).\n")
    w("### 2b. The diffusion 'Normal -> IAC trajectory' is a plotting artefact\n")
    w("```")
    for k, v in A["3_diffusion_trajectory"].items():
        w(f"{k}: {v}")
    w("```")
    w("\n> adjacent/all-pairs distance ratio 0.97 means the 100 points are independent draws")
    w("> from interpolated condition embeddings; joining them with a line is not a path.")
    w("> Generated points sit ~2.8x further from real cells than real cells sit from each other.\n")
    w("### 2c. Generative/interpretive chain after touching the encoder\n")
    w("```")
    for k, v in A["4_recon_corr"].items():
        w(f"{k}: {v}")
    w("```")
    w(f"\n> {A.get('4d_base_is_not_a_lung_cancer_model','')}\n")

# ------------------------------------------------------------ 3-5. ours --
h = J(f"{RES}/finetune_unsup/history.json")
if h:
    H = pd.DataFrame(h)
    w("## 3. This round's unsupervised adaptation: train/val curves\n")
    w("```")
    w(H[["epoch", "train_c", "train_rec", "val_c", "val_rec", "val_recon_corr",
         "eff_rank_val", "mean_norm_val"]].round(3).to_string(index=False))
    w("```")
    w("\n> `val_c` is the contrastive loss on a FROZEN zero-shot kNN graph, so it measures")
    w("> deviation from the global manifold, not fit quality. It must not select checkpoints")
    w("> (using it degenerates to 'train nothing'). Selection uses held-out masked-gene")
    w("> recovery instead: label-free and non-degenerate.\n")
s = C(f"{RES}/sweep_metrics.csv")
if len(s):
    w("## 4. Epoch sweep: adaptation vs manifold preservation\n")
    cols = [c for c in ["model", "mask_recovery_corr_luadTok", "mask_recovery_rmse_luadTok",
                        "recon_corr", "mean_norm"] if c in s.columns]
    w("```"); w(s[cols].to_string(index=False)); w("```")
    b = s.sort_values("mask_recovery_corr_luadTok").iloc[-1]
    w(f"\n>>> best epoch by held-out masked-gene recovery: **{b['model']}**\n")
files = sorted(glob.glob(f"{RES}/comparison_*.csv"))
if files:
    c = pd.concat([C(f) for f in files], ignore_index=True).drop_duplicates(subset=["model"])
    w("## 5. Head-to-head comparison (everything on held-out donors)\n")
    keep = [x for x in ["model", "mask_recovery_corr_luadTok", "recon_corr",
                        "LOPO_malignant_vs_AT2__GSE148071", "LOPO_celltype6",
                        "LOPO_stage6_CONFOUNDED", "sil_stage6_leaky", "CKA_vs_ZS",
                        "knn20_overlap_vs_ZS", "eff_rank", "mean_norm"] if x in c.columns]
    w("```"); w(c[keep].to_string(index=False)); w("```")
    w("\n> `sil_stage6_leaky` is the class of metric the previous round used as evidence (the")
    w("> same labels built the negatives and then scored the result); it is kept only to show")
    w("> it moves opposite to the honest columns. CKA is insensitive to local re-arrangement")
    w("> (the leaky models still score 0.96); kNN20 overlap is the better preservation measure.\n")
M = J(f"{RES}/marker_benchmark.json")
if M:
    w("## 6. Held-out-donor benchmark that does not depend on the clustering annotation\n")
    w(f"- AT2 panel: {', '.join(M['panel_AT2'])}")
    w(f"- airway/tumour panel: {', '.join(M['panel_tumor'])}")
    w(f"- within-subject paired donors: {M['n_paired_subjects']}\n")
    for key, nm in [("i", "(i) GSE148071 cross-donor: annotated malignant vs normal AT2"),
                    ("ii", "(ii) GSE131907 within-subject: adjacent-normal vs tumour side")]:
        if key in M:
            w(f"**{nm}** -- leave-one-donor-out AUROC")
            w("```"); w(pd.DataFrame(M[key]).to_string(index=False)); w("```")
    w("\n> The three embedding spaces are within each other's standard deviation, i.e. the")
    w("> fine-tuning read no extra biological signal. Reference row `[ref] marker panel`")
    w("> scores only 0.59 in (i): differentiation markers are near-orthogonal to a CNV-based")
    w("> malignancy call (LUAD retains the AT2 program), so that row is a statement about the")
    w("> panel, not proof that the annotation is broken.\n")

P = J(f"{RES}/paired_axis.json")
if P:
    w("## 7. Donor-level identifiability and the within-subject paired axis\n")
    w("```"); w(json.dumps(P.get("identifiability", {}), ensure_ascii=False, indent=1)); w("```")
    if "B_paired_axis" in P:
        w("\nPaired axis (adjacent-normal AT2 -> same-donor intra-tumour AT2; donor effect removed):")
        w("```"); w(pd.DataFrame(P["B_paired_axis"]).T.to_string()); w("```")
        w("\n> `mean_resultant_R` = cross-donor direction consistency; `LOPO_subject_rank_acc`")
        w("> = fraction of unseen donors ordered correctly by an axis learned from the others")
        w("> (chance 0.5). All three spaces tie at 0.889: the axis already exists zero-shot,")
        w("> adaptation only stretched it (axis_norm 2.57 -> 4.06) without improving it.\n")
    d = C(f"{RES}/donor_level_de_GSE148071.csv")
    if len(d):
        w("**Donor-level DE (GSE148071, unit = donor)** -- top 12 by |t|:")
        w("```")
        w(d.reindex(d["welch_t"].abs().sort_values(ascending=False).index)
          .head(12)[["symbol", "mean_logFC", "welch_t", "p_welch", "q_welch"]].round(4).to_string(index=False))
        w("```")
        w("\n> Only 5 control donors, and the top hits are MHC-II / macrophage background genes")
        w("> (HLA-DRA, HLA-DRB1, CYBA): still confounded by composition and donor, so unusable")
        w("> as malignancy markers.\n")

# ------------------------------------------------------------ 8. niches --
if os.path.exists(f"{RES}/niches/niche_summary.json"):
    S = J(f"{RES}/niches/niche_summary.json")
    ad = C(f"{RES}/niches/archetype_definitions.csv", index_col=0)
    cor = C(f"{RES}/niches/old_vs_new_spearman.csv", index_col=0)
    groups = [g for g in ad.columns if g != "n_spots"]
    rows = [r for r in cor.index if not r.startswith("N_")]
    w("## 8. Spatial niches rebuilt from SCMG's own mapping (no clustering labels)\n")
    w(f"- {S['n_slides']} slides / {S['n_spots']:,} spots / **only 10 donors**; {len(groups)} model-native identities")
    w("- within-slide k=6 clustering + cross-slide archetype matching (so clusters do not just")
    w("  recover 'which slide')")
    w(f"- **ablation, identical algorithm, epithelium relabelled back to the old cluster labels:")
    w(f"  ARI = {S['ari_native_vs_oldepi_ablation']:.3f}** -> the niche partition is decided")
    w("  almost entirely by how the epithelium is labelled, and section 9 shows that step is wrong.\n")
    w("### 8a. The 6 old hand-written niche scores vs model-native composition (most robust)\n")
    w("```"); w(cor.loc[rows, [c for c in cor.columns if not c.startswith('archetype')]]
            .astype(float).round(2).to_string()); w("```")
    w("\n- Four old niches survive under native labels with consistent sign: Lepidic <-> normal")
    w("  epithelium +0.78; myCAF barrier <-> pericyte/smooth muscle/fibroblast +0.95..0.98;")
    w("  immune hotspot <-> T +0.90 (NK +0.88, B +0.84); TAM barrier <-> macrophage +0.90.")
    w("- `Invasive_Solid_Core_Niche` anchors on `Other/Embryonic` (+0.97). That bucket is")
    w("  dominated by the reference's epidermal/basal epithelial program (13,133 cells), which")
    w("  is a *plausible* read for solid/squamoid LUAD, not a junk bin -- but it also contains")
    w("  8,575 cells annotated normal AT2, so its real meaning is 'epithelium with no normal")
    w("  lung counterpart'.")
    w("- `Exhausted_TME_Niche` is -0.89 with normal epithelium: it is effectively a synonym for")
    w("  'little normal epithelium' and carries no T-cell-exhaustion information. The name")
    w("  over-interprets the quantity.")
    w("- **Boundary: `argmax` native cell type is not trustworthy for tumour cells** (the")
    w("  reference has no tumour, so they get snapped to prostate / thymic / tooth epithelium).")
    w("  The usable quantity is the distance, section 9.\n")
    w("### 8b. Prototype level (discrete clusters)\n")
    w("```"); w(cor.loc[rows, [c for c in cor.columns if c.startswith('archetype')]]
            .astype(float).round(2).to_string()); w("```")
    w("\n> Correspondence is much weaker than at composition level (max 0.75; Lepidic and")
    w("> immune hotspot only 0.28-0.35): the old 'niches' are linear combinations of")
    w("> composition, not sharply separated spatial entities.\n")
    w("### 8c. Two data defects that must be fixed first\n")
    w("1. `all_slides_spatial_deconv_master.csv` contains only the 7 GSE307534 slides, while the")
    w("   old niche table reports 15; the other 8 never entered the master (rebuilt here as")
    w("   15 slides / 118,565 spots / 10 donors).")
    w("2. **5 of 15 slides carry a `stage` that contradicts the ground-truth manifest**, all in")
    w("   GSE189487: TD1/TD2 labelled AIS are IAC, TD5 labelled MIA is AIS, TD6 labelled IAC is")
    w("   MIA, TD8 labelled IAC is AIS. The x-axis of the stage-niche curves was wrong.\n")
    w("### 8d. Method note: the old 6 'niches' were never clustered\n")
    w("`12_spatial_cellular_neighborhoods_and_niches.py` builds them as hand-written linear")
    w("combinations of Seurat-transfer probabilities (`lepidic_score`, `solid_score`,")
    w("`mycaf_score`, `tex_score`). Their high correlation with native composition is therefore")
    w("the same object written twice, **not** independent validation.\n")
    w("### 8e. Strongest design, and it is negative\n")
    w("The only properly paired spatial design is GSE307534 multi-region per subject:")
    w("Patient_1 (AAH vs IAC), Patient_2 (AAH vs IAC), Patient_3 (AIS vs IAC) -> **n = 3 donors**.")
    w("Across those pairs every native compartment and every archetype fraction shifts by")
    w("|delta| <= 0.018 with inconsistent signs -> **no niche changes reproducibly from")
    w("pre-invasive to invasive within a donor.** Yet the epithelial abnormality score for the")
    w("same donors is 9/9 consistent, p = 0.004 (section 9). **What changes is the epithelial")
    w("cell itself, not its niche.** That is the most important result of the niche work.\n")
    w("![niche](niches/niche_comparison.png)\n")

# ---------------------------------------------------- 9. epithelium call --
EN, ec = J(f"{RES}/model_native/epithelium_novelty.json"), C(f"{RES}/model_native/epithelium_calls.csv")
if EN and len(ec):
    t95 = float(np.quantile(ec[(ec.dataset == "GSE131907") & (ec.site == "adjacent_normal")].d_to_normal_epithelium, .95))
    w("## 9. Model-native epithelial abnormality -- a third judge, using neither annotation nor CNV\n")
    w("Raw novelty to the global reference is unusable (it is dominated by reference class")
    w("density: IgA+ plasma 5.3 and myCAF 2.8 exceed most tumour clones). Restricted instead to")
    w("the reference's normal respiratory/epithelial cells and density-corrected by the")
    w(f"reference's own 1NN distances, with the threshold set at the adjacent-normal q95 = {t95:.2f}:\n")
    w("```")
    w(ec[ec.dataset == "GSE131907"].groupby("site").agg(med_dist=("d_to_normal_epithelium", "median"),
      frac_abnormal=("abn95", "mean"), n=("abn95", "size"), donors=("patient_id", "nunique"))
      .round(3).to_string())
    w("```")
    w(f"\n**Within-donor paired test, GSE131907 ({EN['n_paired_subjects']} donors; every one of these")
    w("epithelial cells is annotated 'Normal Alveolar Epithelial')**\n")
    w("```")
    w(f"tumour side farther in {EN['frac_tumor_farther']:.2f} of donors | mean paired delta = {EN['paired_delta_mean']:.3f}")
    w(f"paired t p = {EN['paired_t_p']:.4g} | Wilcoxon p = {EN['paired_wilcoxon_p']:.4g}")
    w(f"within-subject AUC = {EN['auc_within_subject_mean']:.3f} (pooled {EN['auc_pooled']:.3f})")
    w("```")
    w("\n1. **This independently reproduces the aneuploidy result.** Without looking at any")
    w("   annotation or CNV output, 9/9 donors' tumour-sample epithelium sits significantly")
    w("   farther from the normal lung epithelial manifold than the same donor's adjacent")
    w("   normal. 'GSE131907 contains zero malignant epithelium' is therefore an annotation")
    w("   error, not an absence of tumour.")
    w("2. Within-dataset gradient: adjacent normal 5% -> primary tumour 19% -> lymph-node")
    w("   metastasis 46% -> pleural effusion 56% called abnormal.")
    w("3. **Not comparable across datasets**: the same annotated malignant epithelium medians")
    w("   0.96 in GSE189357 vs 2.69 in GSE148071. Platform/handling differences swamp")
    w("   biology, so this judge is only valid within a dataset and within a donor.")
    w("4. Final division of labour is unchanged: **compartments from SCMG's global mapping,")
    w("   malignancy from CNV, this distance as an independent cross-check.** When the CNV")
    w("   table arrives (barcode, score, call, method, reference-cell definition), compute")
    w("   novelty-vs-CNV AUROC directly.\n")

open(f"{RES}/REPORT.md", "w").write("\n".join(md) + "\n")
print(f"wrote {RES}/REPORT.md ({len(md)} lines)")
