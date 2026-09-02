#!/usr/bin/env python
"""83: Re-build the malignant-epithelium call for GSE131907 from CNV
evidence, replacing the fine labels that claim 0 malignant epithelium
(README section 7.9 / 8). Two orthogonal annotation-free channels:
  - CNV load (71): cnv_call at the dataset q99 matched-null threshold
  - epithelium novelty (61/62): abn95 vs cancer-adjacent normal q95
Combined call: 'cnv_novel_both' > 'single' > 'neither'.
Rests on the self-run CNV implementation pending 82 endorsement.

Outputs (results/cnv/GSE131907/):
  epithelium_cnv_calls.csv   per-cell two-channel table + combined call
  donor_cnv_calls.csv        donor x site summary
  cnv_recall_summary.json
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RES = '/home/ubuntu/luad_scmg_v2/results'


def main():
    nov = pd.read_csv(f'{RES}/model_native/epithelium_calls.csv')
    nov = nov[nov['dataset'] == 'GSE131907']
    cnv = pd.read_csv(f'{RES}/cnv/GSE131907/cnv_per_cell.csv')
    thr = json.load(open(f'{RES}/cnv/GSE131907/cnv_summary.json'))
    thr_q99 = thr['thr_q99_matched_null']

    df = nov.merge(
        cnv[['barcode', 'cnv_load', 'cnv_call', 'is_null_holdout']],
        on='barcode', how='inner', validate='1:1')
    test = df[~df['is_null_holdout'].astype(bool)].copy()

    # combined two-channel call
    test['cnv_abn'] = test['cnv_load'] > thr_q99
    test['both'] = (test['cnv_abn'] & (test['abn95'] == 1)).astype(int)
    test['either'] = (test['cnv_abn'] | (test['abn95'] == 1)).astype(int)
    test['malignant_call_cnv'] = np.select(
        [test['both'] == 1, test['either'] == 1],
        ['cnv_novel_both', 'single_evidence'], default='neither')

    cols = ['barcode', 'patient_id', 'site', 'stage', 'compartment',
            'z_novelty', 'abn95', 'cnv_load', 'cnv_call', 'cnv_abn',
            'malignant_call_cnv']
    test[cols].to_csv(f'{RES}/cnv/GSE131907/epithelium_cnv_calls.csv',
                      index=False)

    summ = {
        'threshold_cnv_q99': thr_q99,
        'n_epithelium_test': int(len(test)),
        'frac_cnv_abn': float(test['cnv_abn'].mean()),
        'frac_novel_abn95': float((test['abn95'] == 1).mean()),
        'frac_both': float(test['both'].mean()),
        'frac_either': float(test['either'].mean()),
        'call_counts': test['malignant_call_cnv'].value_counts().to_dict(),
    }

    # donor-level: every donor, both channels
    donor = test.groupby(['patient_id', 'site']).agg(
        n=('barcode', 'size'),
        frac_cnv=('cnv_abn', 'mean'),
        frac_novel=('abn95', 'mean'),
        frac_both=('both', 'mean'),
        med_cnv=('cnv_load', 'median'),
        med_nov=('z_novelty', 'median')).reset_index()
    donor = donor.sort_values('frac_cnv', ascending=False)
    donor.to_csv(f'{RES}/cnv/GSE131907/donor_cnv_calls.csv', index=False)
    summ['n_donors'] = int(len(donor))
    summ['donors_with_both_gt_5pct'] = int((donor['frac_both'] > 0.05).sum())
    summ['donors_with_cnv_gt_5pct'] = int((donor['frac_cnv'] > 0.05).sum())

    # Spearman between channels across donors (gradient-level agreement)
    rho, p = stats.spearmanr(donor['frac_cnv'], donor['frac_novel'])
    summ['donor_level_spearman_cnv_vs_novel'] = [round(float(rho), 4),
                                                 float(p)]

    with open(f'{RES}/cnv/GSE131907/cnv_recall_summary.json', 'w') as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    print(donor.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
