"""
Download the trained MolmoAct2 checkpoint from RunPod's S3-compatible storage.

Requires RUNPOD_S3_ACCESS_KEY_ID and RUNPOD_S3_SECRET_ACCESS_KEY set in the
environment (RunPod dashboard -> S3 API keys). Resumable: skips any file
already present at the destination with a matching size.

Usage: python download_molmoact2_checkpoint.py [dest_dir]
"""
import os
import sys
import time

import boto3

DEST_DIR = sys.argv[1] if len(sys.argv) > 1 else "./mira_molmoact2_step2000"

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3api-us-ca-2.runpod.io/",
    aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
    region_name="us-ca-2",
)

bucket = "fpvqvxh8e0"
prefix = "mira_molmoact2_training/outputs/mira-so101-molmoact2-canary/checkpoints/002000/pretrained_model/"

os.makedirs(DEST_DIR, exist_ok=True)

paginator = s3.get_paginator("list_objects_v2")
keys = []
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        keys.append((obj["Key"], obj["Size"]))

CHUNK = 8 * 1024 * 1024

for key, size in keys:
    fname = key[len(prefix):]
    dest_path = os.path.join(DEST_DIR, fname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == size:
        print(f"SKIP (already correct size): {fname}", flush=True)
        continue
    print(f"Downloading {fname} ({size/1024/1024:.1f} MB) via streamed GET...", flush=True)
    t0 = time.perf_counter()
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    written = 0
    last_print = time.perf_counter()
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as f:
        while True:
            chunk = body.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            now = time.perf_counter()
            if now - last_print > 5:
                elapsed = now - t0
                mbps = (written / 1024 / 1024) / elapsed if elapsed > 0 else 0
                print(f"  {100*written/size:.1f}% ({written/1024/1024:.0f}/{size/1024/1024:.0f} MB) {mbps:.1f} MB/s", flush=True)
                last_print = now
    os.replace(tmp_path, dest_path)
    elapsed = time.perf_counter() - t0
    mb = size / 1024 / 1024
    print(f"  done in {elapsed:.1f}s ({mb/elapsed:.1f} MB/s)", flush=True)

print("\nAll files downloaded to", DEST_DIR, flush=True)
