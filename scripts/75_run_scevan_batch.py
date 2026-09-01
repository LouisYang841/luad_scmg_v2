#!/usr/bin/env python3
"""Run independent per-sample SCEVAN jobs with bounded CPU parallelism."""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time

import luadlib as L

parser = argparse.ArgumentParser(description="Run all prepared SCEVAN samples")
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--cores-per-job", type=int, default=4)
parser.add_argument("--timeout-minutes", type=int, default=45)
args = parser.parse_args()

input_root = f"{L.RES}/scevan/input"
output_root = f"{L.RES}/scevan/output"
log_root = f"{L.ROOT}/logs/scevan"
os.makedirs(output_root, exist_ok=True)
os.makedirs(log_root, exist_ok=True)
patients = sorted(
    [name for name in os.listdir(input_root) if os.path.isfile(f"{input_root}/{name}/matrix.mtx.gz")],
    key=lambda value: int(value.rsplit("P", 1)[1]),
)
if not patients:
    raise RuntimeError("no prepared SCEVAN inputs found")

env = os.environ.copy()
env["R_LIBS_USER"] = f"{L.ROOT}/ref/R_libs"
script = f"{L.ROOT}/scripts/74_run_scevan.R"
status_path = f"{L.RES}/scevan/batch_status.json"


def run_one(patient):
    input_dir = f"{input_root}/{patient}"
    output_dir = f"{output_root}/{patient}"
    prediction = f"{output_dir}/predictions.csv"
    if os.path.exists(prediction):
        return patient, {"status": "skipped_complete", "returncode": 0, "elapsed_seconds": 0.0}
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    command = ["Rscript", script, input_dir, output_dir, str(args.cores_per_job)]
    started = time.time()
    log_path = f"{log_root}/{patient}.log"
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                command,
                cwd=L.ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_minutes * 60,
                check=False,
            )
        state = "complete" if proc.returncode == 0 and os.path.exists(prediction) else "failed"
        return patient, {
            "status": state,
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.time() - started, 1),
            "log": os.path.relpath(log_path, L.ROOT),
        }
    except subprocess.TimeoutExpired:
        return patient, {
            "status": "timeout",
            "returncode": None,
            "elapsed_seconds": round(time.time() - started, 1),
            "log": os.path.relpath(log_path, L.ROOT),
        }


results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
    futures = {pool.submit(run_one, patient): patient for patient in patients}
    for future in concurrent.futures.as_completed(futures):
        patient, result = future.result()
        results[patient] = result
        print(f"{patient}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
        with open(f"{status_path}.tmp", "w") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        os.replace(f"{status_path}.tmp", status_path)

counts = {}
for result in results.values():
    counts[result["status"]] = counts.get(result["status"], 0) + 1
print(json.dumps(counts, indent=2, sort_keys=True))
