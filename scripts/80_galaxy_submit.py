#!/usr/bin/env python3
"""80_galaxy_submit.py -- fire-and-forget inferCNV cross-validation on Galaxy Europe.

Builds the two Galaxy-format helper files from project results (no header, tab-separated):
  gene_order.tsv      gene | chr | start | end   (protein_coding, chr1-22,X, position-sorted)
  cell_annotations.tsv barcode | group           (group: "normal_reference" for P1-P5 normal epi,
                                                 "malignant_<patient>" otherwise)
Then creates a history, uploads the 10x triplet + helpers, waits for the uploads to
enter the 'ok' state, and submits infercnv/1.26.0+galaxy0 with ref_group=normal_reference.
Everything after submission happens on Galaxy's servers; job id is written to
results/galaxy_infercnv/submission.json for later polling.
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


def build_helper_files():
    pos = pd.read_csv(f"{L.RES}/gene_positions_gencode44.csv")
    pos = pos[pos.gene_type == "protein_coding"].copy()
    pos = pos[pos.chrom.isin([f"chr{c}" for c in list(range(1, 23)) + ["X"]])]
    pos["cnum"] = pos.chrom.str[3:].map({str(c): i for i, c in enumerate(list(range(1, 23)) + ["X"])})
    pos = pos.sort_values(["cnum", "start"])
    go = f"{OUT}/gene_order.tsv"
    pos[["ensg", "chrom", "start", "end"]].to_csv(go, sep="\t", index=False, header=False)
    print(f"gene_order.tsv: {len(pos):,} genes", flush=True)

    meta = pd.read_csv(f"{SRC}/metadata.csv", dtype=str)
    meta["group"] = meta.apply(
        lambda r: "normal_reference" if r.is_reference == "True"
        else f"{r.compartment_clean}_{r.patient_id}", axis=1)
    ca = f"{OUT}/cell_annotations.tsv"
    meta[["barcode", "group"]].to_csv(ca, sep="\t", index=False, header=False)
    print(f"cell_annotations.tsv: {len(meta):,} cells, "
          f"{(meta.group == 'normal_reference').sum():,} reference", flush=True)
    return go, ca


def wait_dataset(ds_id, label, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = S.get(f"{API}/datasets/{ds_id}").json().get("state")
        if st == "ok":
            print(f"  uploaded {label}: {ds_id}", flush=True)
            return ds_id
        if st in ("error", "discarded"):
            raise RuntimeError(f"upload failed for {label}: state={st}")
        time.sleep(10)
    raise TimeoutError(f"upload timeout for {label}")


def upload(history_id, path, label):
    with open(path, "rb") as fh:
        r = S.post(f"{API}/tools", data={
            "tool_id": "upload1", "history_id": history_id,
            "inputs": json.dumps({"files_0|file_type": "auto", "files_0|dbkey": "?",
                                  "files_0|NAME": os.path.basename(path)}),
        }, files={"files_0|file_data": fh})
    r.raise_for_status()
    out = r.json()["outputs"][0]
    return wait_dataset(out["id"], label)


def main():
    go, ca = build_helper_files()
    h = S.post(f"{API}/histories", json={"name": "luad_GSE148071_infercnv"}).json()
    hid = h["id"]
    print(f"history: {h.get('name')} {hid}", flush=True)

    mtx = upload(hid, f"{SRC}/matrix.mtx.gz", "matrix.mtx.gz")
    genes = upload(hid, f"{SRC}/genes.tsv", "genes.tsv")
    barcodes = upload(hid, f"{SRC}/barcodes.tsv", "barcodes.tsv")
    go_id = upload(hid, go, "gene_order.tsv")
    ca_id = upload(hid, ca, "cell_annotations.tsv")

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
    out_ids = {o.get("output_name", o.get("id")): o["id"] for o in resp.get("outputs", [])}
    result = {"history_id": hid, "job_ids": jobs, "outputs": out_ids, "tool": TOOL}
    with open(f"{OUT}/submission.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"job submitted: {jobs}", flush=True)


if __name__ == "__main__":
    main()
