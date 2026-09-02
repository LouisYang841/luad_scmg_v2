#!/usr/bin/env python
"""86 (v2): Galaxy EU inferCNV outage retry loop.

The tool began failing ~90s into every run on 2026-09-02 morning (3/3
attempts including a control with run#1's original inputs -- platform-side
issue, all local variables ruled out).  /api/jobs/{id}/rerun does not exist
in Galaxy 26.1, so resubmit the tool against the ALREADY-UPLOADED datasets
in the v2 history (no re-upload).  Retry every 30 min until a submission
survives; then run 85 to compute the official AUROC.

Usage: nohup python 86_galaxy_retry_loop.py > ../logs/86_retry_loop.log 2>&1 &
"""
import json
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luadlib as L

API = "https://usegalaxy.eu/api"
KEY = os.environ.get("GALAXY_API_KEY", "21f4cdfbcc84ee32913470f16e02212b")
S = requests.Session()
S.headers.update({"X-Api-Key": KEY})
TOOL = "toolshed.g2.bx.psu.edu/repos/iuc/infercnv/infercnv/1.26.0+galaxy0"
OUT = f"{L.RES}/galaxy_infercnv"
SUB = f"{OUT}/submission_v2.json"
MAX_ATTEMPTS = 12
SETTLE_SEC = 600


def job_state(jid):
    return S.get(f"{API}/jobs/{jid}").json().get("state")


def resubmit(hid):
    """Resubmit inferCNV reusing the five already-uploaded input datasets."""
    contents = S.get(f"{API}/histories/{hid}/contents").json()
    ids = {}
    for c in contents:
        if c["history_content_type"] != "dataset" or c.get("state") != "ok":
            continue
        ids[c["name"]] = c["id"]
    need = ["matrix.mtx.gz", "genes.tsv", "barcodes.tsv", "gene_order.tsv",
            "cell_annotations_v2.tsv"]
    for n in need:
        assert n in ids, f"missing uploaded dataset: {n}"
    payload = {
        "tool_id": TOOL, "history_id": hid,
        "inputs": {
            "input_mat|format": "mtx",
            "input_mat|mtx": {"id": ids["matrix.mtx.gz"], "src": "hda"},
            "input_mat|mtx_genes": {"id": ids["genes.tsv"], "src": "hda"},
            "input_mat|mtx_barcodes": {"id": ids["barcodes.tsv"], "src": "hda"},
            "annotation|gene_annotation": "local",
            "annotation|gene_order": {"id": ids["gene_order.tsv"], "src": "hda"},
            "cell_annotations": {"id": ids["cell_annotations_v2.tsv"],
                                 "src": "hda"},
            "ref_group": "normal_reference",
        },
    }
    r = S.post(f"{API}/tools", json=payload)
    log(f"  resubmit HTTP {r.status_code}")
    if r.status_code != 200:
        log(f"  body: {r.text[:300]}")
        return None
    return r.json()["jobs"][0]["id"]


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def main():
    sub = json.load(open(SUB))
    hid = sub["history_id"]
    jid = sub["job_ids"][0]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        st = job_state(jid)
        log(f"attempt {attempt}: job {jid} state={st}")
        if st == "ok":
            break
        if st in ("new", "queued", "running", "waiting"):
            while True:
                time.sleep(600)
                st = job_state(jid)
                log(f"  ... {st}")
                if st in ("ok", "error", "deleted"):
                    break
        if st == "ok":
            break
        time.sleep(1800)
        new_jid = resubmit(hid)
        if new_jid:
            jid = new_jid
            sub["job_ids"] = [new_jid]
            json.dump(sub, open(SUB, "w"), indent=2)
    else:
        log("giving up after max attempts")
        return

    log("job OK -- running 85")
    subprocess.run(["/home/ubuntu/qwen-env/bin/python",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "85_galaxy_v2_auroc.py")], check=True)


if __name__ == "__main__":
    main()
