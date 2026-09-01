#!/usr/bin/env python3
"""70_gene_positions.py -- ENSG -> (chrom, start, end) from GENCODE v44 basic GTF.

Needed because the friend's uploaded rds (a) has NO CNV column and (b) is named by
gene SYMBOL, while our own pipeline is named by ENSG. We compute CNV ourselves from
our 18,108-gene standard matrix, so we only need positions for ENSG ids.

Arms: p/q split at the centromere. GRCh38 centromere midpoints are hardcoded below
from the UCSC hg38 cytoBand `acen` interval boundaries so the script has no runtime
network dependency; the midpoint used for every gene is written for transparency.
"""
import os, re, gzip
import numpy as np, pandas as pd
import luadlib as L

GTF = "/home/ubuntu/luad_scmg_v2/ref/gencode.v44.basic.annotation.gtf.gz"
OUT = f"{L.RES}/gene_positions_gencode44.csv"

# GRCh38 centromere midpoint per chromosome (bp), from UCSC hg38 cytoBand `acen`.
CENTRO = {
    "1": 123_400_000, "2": 93_900_000, "3": 90_900_000, "4": 50_000_000,
    "5": 48_750_000, "6": 60_550_000, "7": 60_100_000, "8": 45_200_000,
    "9": 43_850_000, "10": 39_800_000, "11": 53_400_000, "12": 35_500_000,
    "13": 17_700_000, "14": 17_150_000, "15": 19_000_000, "16": 36_850_000,
    "17": 25_050_000, "18": 18_450_000, "19": 26_150_000, "20": 28_050_000,
    "21": 11_950_000, "22": 15_550_000, "X": 60_950_000, "Y": 10_450_000,
}

rows = []
with gzip.open(GTF, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        m = re.search(r'gene_id "([^".]+)', f[8])
        t = re.search(r'gene_type "([^"]+)', f[8])
        if not m:
            continue
        rows.append((m.group(1), f[0], int(f[3]), int(f[4]), t.group(1) if t else "NA"))
g = pd.DataFrame(rows, columns=["ensg", "chrom", "start", "end", "gene_type"]).drop_duplicates("ensg")
g = g[g.chrom.str.match(r"chr[0-9XY]+$")]
g["arm"] = [f"{c[3:]}{'p' if s < CENTRO.get(c[3:], 1e12) else 'q'}" for c, s in zip(g.chrom, g.start)]
g["centromere"] = [CENTRO.get(c[3:], np.nan) for c in g.chrom]
g.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(g)} genes on {g.chrom.nunique()} chromosomes")
print(g.gene_type.value_counts().head(6).to_string())
pc = g[g.gene_type == "protein_coding"]
print(f"protein_coding: {len(pc)}; 覆盖 {pc.ensg.nunique()} ENSG")
