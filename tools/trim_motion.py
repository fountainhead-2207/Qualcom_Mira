"""
Trim frames off the end (or start) of a recorded single-episode motion dataset.

lerobot-edit-dataset can delete whole episodes, split, and merge, but has no
operation for shortening one episode - and re-recording just to drop a second
off the tail wastes a take that was otherwise good.

Keeps the dataset self-consistent, which matters because the board's
load_motion() reads data + info.json while lerobot_replay goes through
LeRobotDataset and also reads meta/episodes + stats.json:
  - truncates data/chunk-000/file-000.parquet and renumbers timestamp,
    frame_index and index so they stay 0-based and contiguous
  - updates info.json total_frames
  - updates meta/episodes length / dataset_to_index
  - recomputes meta/stats.json and the per-episode stats/* columns from the
    surviving frames

Usage:
    python trim_motion.py <dataset_dir> --end 2.0      # drop last 2 seconds
    python trim_motion.py <dataset_dir> --start 0.5    # drop first 0.5 seconds
    python trim_motion.py <dataset_dir> --end 60f      # drop last 60 frames
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


def parse_amount(value, fps):
    """'2.0' -> seconds -> frames; '60f' -> frames."""
    if value is None:
        return 0
    text = str(value).strip()
    if text.endswith("f"):
        return int(text[:-1])
    return int(round(float(text) * fps))


def compute_stats(values):
    """Per-column stats in the layout LeRobot writes (lists, one entry per dim)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [arr.shape[0]],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--end", default=None, help="how much to drop off the end (seconds, or Nf frames)")
    ap.add_argument("--start", default=None, help="how much to drop off the start")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = info["fps"]

    data_path = next((root / "data").glob("chunk-*/*.parquet"))
    table = pq.read_table(data_path)
    total = table.num_rows

    drop_start = parse_amount(args.start, fps)
    drop_end = parse_amount(args.end, fps)
    keep = total - drop_start - drop_end
    if keep < 2:
        raise SystemExit(f"nothing left: {total} frames minus {drop_start}+{drop_end}")

    print(f"{total} frames ({total / fps:.1f}s) -> {keep} frames ({keep / fps:.1f}s)"
          f"  [dropping {drop_start} from start, {drop_end} from end]")

    table = table.slice(drop_start, keep)

    # Renumber the bookkeeping columns so the trimmed episode still starts at 0.
    # LeRobotDataset trusts these to line up with dataset_from_index/to_index.
    table = table.set_column(table.schema.get_field_index("timestamp"), "timestamp",
                             pa.array((np.arange(keep) / fps).astype(np.float32)))
    for col in ("frame_index", "index"):
        table = table.set_column(table.schema.get_field_index(col), col,
                                 pa.array(np.arange(keep, dtype=np.int64)))
    pq.write_table(table, data_path)

    info["total_frames"] = keep
    info_path.write_text(json.dumps(info, indent=4), encoding="utf-8")

    # stats.json (dataset-wide) and the stats/* columns on the episode row both
    # have to reflect only the frames that survived.
    stats = {}
    for name in table.column_names:
        stats[name] = compute_stats(table[name].to_pylist())
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")

    ep_path = next((root / "meta/episodes").glob("chunk-*/*.parquet"))
    ep = pq.read_table(ep_path)
    updates = {"length": pa.array([keep], type=ep.schema.field("length").type),
               "dataset_to_index": pa.array([keep], type=ep.schema.field("dataset_to_index").type),
               "dataset_from_index": pa.array([0], type=ep.schema.field("dataset_from_index").type)}
    for feature, values in stats.items():
        for key in STAT_KEYS:
            col = f"stats/{feature}/{key}"
            if col in ep.column_names:
                updates[col] = pa.array([values[key]], type=ep.schema.field(col).type)
    for col, arr in updates.items():
        ep = ep.set_column(ep.schema.get_field_index(col), col, arr)
    pq.write_table(ep, ep_path)

    print("trimmed and metadata rewritten")


if __name__ == "__main__":
    main()
