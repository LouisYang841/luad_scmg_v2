#!/usr/bin/env python
"""85: When the Galaxy v2 inferCNV job (held-out normals as observations)
finally succeeds, download the per-cell ratio matrix and compute the
official third-party AUROC: malignant (13,412) vs held-out normal
epithelium (4,026) -- the same contrast as 71's 0.7569 positive control.

Reads the history recorded in results/galaxy_infercnv/submission_v2.json
(updated by 86 retry loop).  Output: results/galaxy_v2_auroc.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import requests
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

API = "https://usegalaxy.eu/api"
KEY = os.environ.get("GALAXY_API_KEY", "21f4cdfbcc84ee32913470f16e02212b")
S = requests.Session()
S.headers.update({"X-Api-Key": KEY})
OUT = f"{L.RES}/galaxy_infercnv"
DL = f"{OUT}/download_v2"
os.makedirs(DL, exist_ok=True)


def main():
    sub = json.load(open(f"{OUT}/submission_v2.json"))
    hid = sub["history_id"]
    contents = S.get(f"{API}/histories/{hid}/contents").json()

    # biggest tabular inside the Results collection = observations matrix
    best = None
    for c in contents:
        if c["history_content_type"] != "dataset_collection":
            continue
        col = S.get(f"{API}/dataset_collections/{c['id']}").json()
        if "Results" not in c["name"] or col.get("element_count", 0) == 0:
            continue
        for e in col.get("elements", []):
            o = e.get("object", {})
            if o.get("file_ext") == "tabular" and o.get("state") == "ok":
                if best is None or o["file_size"] > best[1]:
                    best = (o["id"], o["file_size"], e["element_identifier"])
    assert best, "no populated Results collection found"
    ds_id, size, name = best
    print(f"observations matrix: {name} ({size/1e6:.1f} MB, {ds_id})", flush=True)

    path = f"{DL}/observations_v2.tsv"
    if not os.path.exists(path) or os.path.getsize(path) != size:
        with S.get(f"{API}/histories/{hid}/contents/{ds_id}/display",
                   stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    print("downloaded", flush=True)

    mat = pd.read_csv(path, sep=" ", index_col=0, nrows=5)
    cols = mat.columns
    # stream per-row to bound memory: 17,438 x 16,937 fits in ~1.2 GB fp32
    m = pd.read_csv(path, sep=" ", index_col=0)
    m.index = m.index.str.strip('"')
    dev = (m.values - 1.0).astype(np.float32)
    g = pd.DataFrame({"barcode": m.index})
    g["galaxy_absdev"] = np.abs(dev).mean(axis=1)
    g["galaxy_frac_dev"] = (np.abs(dev) > 0.3).mean(axis=1)
    g["row_std"] = dev.std(axis=1)
    del m, dev

    ann = pd.read_csv(f"{OUT}/cell_annotations_v2.tsv", sep="\t",
                      header=None, names=["barcode", "group"])
    ann["class"] = np.where(ann["group"] == "normal_reference", "ref",
                    np.where(ann["group"].str.startswith("normal_holdout"),
                             "normal", "malignant"))
    df = g.merge(ann[["barcode", "class"]], on="barcode", how="inner",
                 validate="1:1")

    mal = df.loc[df["class"] == "malignant", "galaxy_absdev"]
    nor = df.loc[df["class"] == "normal", "galaxy_absdev"]
    allv = np.concatenate([mal.values, nor.values])
    ranks = stats.rankdata(allv)
    u = ranks[:len(mal)].sum() - len(mal) * (len(mal) + 1) / 2.0
    auroc = float(u / (len(mal) * len(nor)))

    out = {
        "n_malignant": int(len(mal)), "n_normal_holdout": int(len(nor)),
        "official_infercnv_absdev_auroc": auroc,
        "self_run_71_positive_control_auroc": 0.7569,
        "median_absdev_malignant": float(mal.median()),
        "median_absdev_normal": float(nor.median()),
        "flat_rows_rowstd_lt_0p02": int((df["row_std"] < 0.02).sum()),
    }
    with open(f"{L.RES}/galaxy_v2_auroc.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
