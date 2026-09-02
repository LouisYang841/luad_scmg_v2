#!/usr/bin/env python
"""81: Concordance between model-native epithelial novelty (61/62) and
self-run CNV load (71) -- two orthogonal, annotation-free evidence channels.

Outputs (results/):
  novelty_vs_cnv.json            all headline numbers
  novelty_vs_cnv_by_site.csv     GSE131907 per-site two-method table
  novelty_vs_cnv_donor.csv       per-donor medians of both methods
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RES = '/home/ubuntu/luad_scmg_v2/results'


def auroc(pos, neg):
    """Mann-Whitney AUROC: P(score_pos > score_neg) + 0.5*ties."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = stats.rankdata(allv)
    rp = ranks[:len(pos)].sum()
    u = rp - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def kappa_phi(t):
    a, b = float(t.iloc[0, 0]), float(t.iloc[0, 1])
    c, d = float(t.iloc[1, 0]), float(t.iloc[1, 1])
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    kap = (po - pe) / (1 - pe) if pe < 1 else np.nan
    num = a * d - b * c
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = num / den if den > 0 else np.nan
    return float(kap), float(phi)


def analyze(ds, nov, cnv):
    df = nov.merge(
        cnv[['barcode', 'cnv_load', 'cnv_amp', 'cnv_del', 'cnv_call',
             'is_null_holdout']],
        on='barcode', how='inner', validate='1:1')
    n_merged, n_nov = len(df), len(nov)

    # null-holdout cells set the CNV threshold -> keep them out of the
    # primary concordance (they are the reference distribution, not test cells)
    test = df[~df['is_null_holdout'].astype(bool)].copy()

    out = {'dataset': ds, 'n_merged': n_merged, 'n_novelty': n_nov,
           'n_test': int(len(test))}

    # continuous concordance
    rho, p = stats.spearmanr(test['z_novelty'], test['cnv_load'])
    out['spearman_znovelty_cnvload'] = [round(float(rho), 4), float(p)]
    out['auroc_cnvload_vs_abn95'] = auroc(
        test.loc[test['abn95'] == 1, 'cnv_load'],
        test.loc[test['abn95'] == 0, 'cnv_load'])
    out['auroc_znovelty_vs_cnvcall'] = auroc(
        test.loc[test['cnv_call'], 'z_novelty'],
        test.loc[~test['cnv_call'], 'z_novelty'])

    # hard-call concordance
    t = pd.crosstab(test['abn95'], test['cnv_call'])
    t = t.reindex(index=[0, 1], columns=[False, True], fill_value=0)
    out['table_abn95_x_cnvcall'] = t.values.tolist()
    kap, phi = kappa_phi(t)
    both = int(t.iloc[1, 1])
    either = int(t.values.sum() - t.iloc[0, 0])
    out['kappa'] = round(kap, 4)
    out['phi'] = round(phi, 4)
    out['both_called'] = both
    out['union_called'] = either
    out['jaccard'] = round(both / either, 4) if either else None

    # per-compartment rates
    comp = test.groupby('compartment').agg(
        n=('barcode', 'size'),
        frac_abn95=('abn95', 'mean'),
        frac_cnv=('cnv_call', 'mean'),
        med_znov=('z_novelty', 'median'),
        med_cnv=('cnv_load', 'median'))
    out['by_compartment'] = json.loads(comp.round(4).to_json(orient='index'))

    # site-level gradient (GSE131907 has the informative sites)
    if ds == 'GSE131907':
        site = test.groupby('site').agg(
            n=('barcode', 'size'),
            frac_abn95=('abn95', 'mean'),
            frac_cnv=('cnv_call', 'mean'),
            med_znov=('z_novelty', 'median'),
            med_cnv=('cnv_load', 'median')).sort_values('frac_cnv')
        out['by_site'] = json.loads(site.round(4).to_json(orient='index'))

        # paired tumor-vs-adjacent-normal by numeric sample suffix
        # (rests on the LUNG_T##/LUNG_N## naming inference -- flagged)
        pid = test['patient_id'].astype(str)
        num = pid.str.extract(r'_(?:LUNG|LN)?_?([TN])(\d+)$')
        test = test.assign(side=num[0], pair=num[1].fillna(''))
        rows = []
        for pair, g in test[test['pair'] != ''].groupby('pair'):
            if set(g['side']) != {'T', 'N'}:
                continue
            tmed = g.loc[g['side'] == 'T', 'cnv_load'].median()
            nmed = g.loc[g['side'] == 'N', 'cnv_load'].median()
            tnz = g.loc[g['side'] == 'T', 'z_novelty'].median()
            nnz = g.loc[g['side'] == 'N', 'z_novelty'].median()
            rows.append({'pair': pair, 'cnv_T': tmed, 'cnv_N': nmed,
                         'nov_T': tnz, 'nov_N': nnz,
                         'cnv_T_gt_N': bool(tmed > nmed),
                         'nov_T_gt_N': bool(tnz > nnz)})
        paired = pd.DataFrame(rows)
        if len(paired):
            n_cnv = int(paired['cnv_T_gt_N'].sum())
            n_nov = int(paired['nov_T_gt_N'].sum())
            out['paired_donors'] = int(len(paired))
            out['paired_T_gt_N_cnv'] = n_cnv
            out['paired_T_gt_N_novelty'] = n_nov
            out['paired_binom_p_cnv'] = float(
                stats.binomtest(n_cnv, len(paired), 0.5,
                                alternative='greater').pvalue)
            out['paired_binom_p_novelty'] = float(
                stats.binomtest(n_nov, len(paired), 0.5,
                                alternative='greater').pvalue)
            paired.to_csv(f'{RES}/novelty_vs_cnv_donor.csv', index=False)
            if 'site' in test:
                site.to_csv(f'{RES}/novelty_vs_cnv_by_site.csv')
    return out


def main():
    nov = pd.read_csv(f'{RES}/model_native/epithelium_calls.csv')
    report = {}
    for ds, path in [('GSE148071', f'{RES}/cnv/GSE148071/cnv_per_cell.csv'),
                     ('GSE131907', f'{RES}/cnv/GSE131907/cnv_per_cell.csv')]:
        cnv = pd.read_csv(path)
        report[ds] = analyze(ds, nov, cnv)
    with open(f'{RES}/novelty_vs_cnv.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
