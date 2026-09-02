#!/usr/bin/env python
"""82: Cross-validate the self-run CNV load (71) against the official
InferCNV per-cell matrix produced on usegalaxy.eu (GSE148071, 13,412
malignant-epithelium observations). The Galaxy matrix holds ratios vs the
reference (normal cells ~1.0), so per-cell CNV magnitude = mean |ratio-1|
and deviation fraction = frac(|ratio-1| > 0.3).

Output: results/galaxy_vs_own_cnv.json
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RES = '/home/ubuntu/luad_scmg_v2/results'
MAT = f'{RES}/galaxy_infercnv/download/26c75dcccb616ac88bcd2a56e8a94302.dl'


def main():
    mat = pd.read_csv(MAT, sep=' ', index_col=0)
    mat.index = mat.index.str.strip('"')

    v = mat.values - 1.0   # ratio -> deviation from reference (=1)
    gal = pd.DataFrame({'barcode': mat.index})
    gal['galaxy_mean'] = mat.mean(axis=1).values           # global shift
    gal['galaxy_absdev'] = np.abs(v).mean(axis=1)          # CNV magnitude
    gal['galaxy_frac_dev'] = (np.abs(v) > 0.3).mean(axis=1)
    gal['row_std'] = v.std(axis=1)                         # CNV shape present?
    del v, mat

    # flat rows (no gene-to-gene variation) carry no CNV information
    sample = gal['barcode'].str.extract(r'^([^_]+)_', expand=False)
    flat = gal.groupby(sample)['row_std'].median()

    own = pd.read_csv(f'{RES}/cnv/GSE148071/cnv_per_cell.csv')
    df = gal.merge(own[['barcode', 'cnv_load', 'cnv_call']],
                   on='barcode', how='inner', validate='1:1')

    out = {'n_matched': len(df), 'n_galaxy': len(gal),
           'n_own_148071': len(own),
           'n_samples_flat_rowstd_lt_0p02': int((flat < 0.02).sum()),
           'flat_samples': sorted(flat[flat < 0.02].index.tolist())}
    for a in ['galaxy_absdev', 'galaxy_frac_dev', 'galaxy_mean']:
        rho, p = stats.spearmanr(df[a], df['cnv_load'])
        r, _ = stats.pearsonr(df[a], df['cnv_load'])
        out[f'spearman_{a}_vs_cnv_load'] = round(float(rho), 4)
        out[f'pearson_{a}_vs_cnv_load'] = round(float(r), 4)
        qa = df[a] > df[a].quantile(0.9)
        qb = df['cnv_load'] > df['cnv_load'].quantile(0.9)
        inter = int((qa & qb).sum())
        out[f'top10pct_overlap_{a}'] = f'{inter}/{int(qa.sum())}'

    with open(f'{RES}/galaxy_vs_own_cnv.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
