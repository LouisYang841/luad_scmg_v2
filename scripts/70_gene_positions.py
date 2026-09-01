#!/usr/bin/env python3
"""70_gene_positions.py -- ENSG -> (chrom, start, end) from GENCODE v44 basic GTF.

Needed because the friend's uploaded rds (a) has NO CNV column and (b) is named by
gene SYMBOL, while our own pipeline is named by ENSG. We compute CNV ourselves from
our 18,108-gene standard matrix, so we only need positions for ENSG ids.

Arms: p/q split at the centromere. GRCh38 centromere midpoints are hardcoded below
(from UCSC cytoBand acrocentric/medacentric boundaries) so we do not depend on a
46-row download at run time; the file is also written for transparency.
"""
import os, re, gzip
import numpy as np, pandas as pd
import luadlib as L

GTF = "/home/ubuntu/luad_scmg_v2/ref/gencode.v44.basic.annotation.gtf.gz"
OUT = f"{L.RES}/gene_positions_gencode44.csv"

# GRCh38 centromere midpoint per chromosome (bp). Used only to split p/q arms.
CENTRO = {1:123471,2:93938,3:90502,4:500000,5:488000,6:598292,7:606000,8:451693,
          9:430411,10:398112,11:534055,12:356000,13:177396,14:172000,15:190000,
          16:366000,17:251000,18:185000,19:263196,20:280000,21:120710,22:50000,
          "X":61000000,"Y":10400000}

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
