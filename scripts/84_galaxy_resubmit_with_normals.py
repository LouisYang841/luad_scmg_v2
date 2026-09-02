#!/usr/bin/env python
"""84: Resubmit inferCNV on usegalaxy.eu with the held-out normal epithelium
moved into the observations.  Run #1 put all 8,016 normal cells into the
reference, so the official output had no normal contrast and could not yield
a third-party AUROC.  Here the reference is the same 3,990 non-holdout
normal-epithelium cells that 71 used, and the 4,026 held-out normals join
the 13,412 malignant observations.  The official malignant-vs-heldout AUROC
then measures the same contrast as 71's 0.7569 positive control.

Reuses input/ (matrix.mtx.gz, genes.tsv, barcodes.tsv) unchanged; only
cell_annotations.tsv is rebuilt.  Writes submission_v2.json.

Output: results/galaxy_infercnv/submission_v2.json
"""
import json
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

API = "https://usegalaxy.eu/api"
KEY = os.environ.get("GALAXY_API_KEY", "21f4cdfbcc84ee32913470f16e02212b")
HDR = {"X-Api-Key": KEY}
TOOL = "toolshed.g2.bx.psu.edu/repos/iuc/infercnv/infercnv/1.26.0+galaxy0"
SRC = f"{L.RES}/galaxy_infercnv/input"
OUT = f"{L.RES}/galaxy_infercnv"

S = requests.Session()
S.headers.update(HDR)


def build_annotations():
    meta = pd.read_csv(f"{SRC}/metadata.csv", dtype=str)
    cnv = pd.read_csv(f"{L.RES}/cnv/GSE148071/cnv_per_cell.csv",
                      usecols=["barcode", "is_null_holdout"])
    hold = dict(zip(cnv["barcode"], cnv["is_null_holdout"].astype(str) == "True"))
    meta["holdout"] = meta["barcode"].map(hold).fillna(False)

    def grp(r):
        if r.is_reference == "True" and not r.holdout:
            return "normal_reference"
        if r.is_reference == "True":          # held-out normal -> observation
            return f"normal_holdout_{r.patient_id}"
        return f"malignant_{r.patient_id}"

    meta["group"] = meta.apply(grp, axis=1)
    ca = f"{OUT}/cell_annotations_v2.tsv"
    meta[["barcode", "group"]].to_csv(ca, sep="\t", index=False,
                                      header=False)
    vc = meta["group"].str.replace(r"_GSE148071_.*", "", regex=True)
    print(f"cell_annotations_v2.tsv: {len(meta):,} cells -> "
          f"{vc.value_counts().to_dict()}", flush=True)
    return ca


def upload(history_id, path, label, timeout=1800):
    with open(path, "rb") as fh:
        r = S.post(f"{API}/tools", data={
            "tool_id": "upload1", "history_id": history_id,
            "inputs": json.dumps({"files_0|file_type": "auto",
                                  "files_0|dbkey": "?",
                                  "files_0|NAME": os.path.basename(path)}),
        }, files={"files_0|file_data": fh})
    r.raise_for_status()
    out = r.json()["outputs"][0]
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = S.get(f"{API}/datasets/{out['id']}").json().get("state")
        if st == "ok":
            print(f"  uploaded {label}: {out['id']}", flush=True)
            return out["id"]
        if st in ("error", "discarded"):
            raise RuntimeError(f"upload failed for {label}: state={st}")
        time.sleep(10)
    raise TimeoutError(f"upload timeout for {label}")


def main():
    ca = build_annotations()
    go = f"{OUT}/gene_order.tsv"          # rebuilt in run #1, unchanged
    h = S.post(f"{API}/histories",
               json={"name": "luad_GSE148071_infercnv_v2"}).json()
    hid = h["id"]
    print(f"history: {h.get('name')} {hid}", flush=True)

    mtx = upload(hid, f"{SRC}/matrix.mtx.gz", "matrix.mtx.gz")
    genes = upload(hid, f"{SRC}/genes.tsv", "genes.tsv")
    barcodes = upload(hid, f"{SRC}/barcodes.tsv", "barcodes.tsv")
    go_id = upload(hid, go, "gene_order.tsv")
    ca_id = upload(hid, ca, "cell_annotations_v2.tsv")

    payload = {
        "tool_id": TOOL, "history_id": hid,
        "inputs": {
            "input_mat|format": "mtx",
            "input_mat|mtx": {"id": mtx, "src": "hda"},
            "input_mat|mtx_genes": {"id": genes, "src": "hda"},
            "input_mat|mtx_barcodes": {"id": barcodes, "src": "hda"},
            "annotation|gene_annotation": "local",
            "annotation|gene_order": {"id": go_id, "src": "hda"},
            "cell_annotations": {"id": ca_id, "src": "hda"},
            "ref_group": "normal_reference",
        },
    }
    r = S.post(f"{API}/tools", json=payload)
    print(f"submit HTTP {r.status_code}", flush=True)
    if r.status_code != 200:
        print(r.text[:3000], flush=True)
        raise SystemExit(1)
    resp = r.json()
    jobs = [j["id"] for j in resp.get("jobs", [])]
    out_ids = {o.get("output_name", o.get("id")): o["id"]
               for o in resp.get("outputs", [])}
    result = {"history_id": hid, "job_ids": jobs, "outputs": out_ids,
              "tool": TOOL, "purpose": "v2: held-out normals as observations"}
    with open(f"{OUT}/submission_v2.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"job submitted: {jobs}", flush=True)


if __name__ == "__main__":
    main()
