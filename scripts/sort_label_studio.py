#!/usr/bin/env python3
# ovo je file koji sortira audio fileove iz label studio batcha u foldere po labelima, npr. CAR, MC, BIC, HEAVY, NOISE, TRAM
# nebitno za daljnju analizu, samo pomocni file
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Optional


DEFAULT_JSON_DIR = Path("data/label_json")
DEFAULT_SRC_DIR = Path("data/sorted_by_vehicle/UNKNOWN")
DEFAULT_DEST_DIR = Path("data/sorted_label_studio")


def map_label_to_category(label: Optional[str]) -> str:
    if not label:
        return "NOISE"
    l = label.lower()
    if "tram" in l:
        return "TRAM"
    if "car" in l and "carriage" not in l:
        return "CAR"
    if any(k in l for k in ("motor", "mc", "motorcycle", "moped", "bike-motor")):
        return "MC"
    if any(k in l for k in ("bicycle", "bicy", "bike")):
        return "BIC"
    if any(k in l for k in ("bus", "truck", "heavy")):
        return "HEAVY"
    if any(k in l for k in ("noise", "unrecognized", "unknown")):
        return "NOISE"
    # fallback
    return "NOISE"


def find_file_by_basename(root: Path, basename: str) -> Optional[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == basename:
                return Path(dirpath) / fn
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sort Label Studio audio by label into categories")
    p.add_argument("batch", help="JSON filename under data/label_json (e.g. batch2.json) or full path")
    p.add_argument("--src-dir", default=str(DEFAULT_SRC_DIR), help="Source folder to search audio files")
    p.add_argument("--dest-dir", default=str(DEFAULT_DEST_DIR), help="Destination base folder")
    p.add_argument("--json-dir", default=str(DEFAULT_JSON_DIR), help="Folder where batch JSONs live")
    p.add_argument("--apply", action="store_true", help="Actually copy files (default is dry-run)")
    p.add_argument("--report", help="Write JSON report to given path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    batch_path = Path(args.batch)
    if not batch_path.is_absolute():
        batch_path = Path(args.json_dir) / batch_path
    if not batch_path.suffix:
        batch_path = batch_path.with_suffix('.json')
    if not batch_path.exists():
        print(f"ERROR: batch JSON not found: {batch_path}")
        return 2

    with open(batch_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    src_root = Path(args.src_dir)
    dest_root = Path(args.dest_dir)
    stats = {"total": 0, "copied": 0, "missing": 0}
    missing = []
    actions = []

    for it in items:
        stats["total"] += 1
        audio_path = None
        data = it.get("data") or {}
        audio_path = data.get("audio")
        if not audio_path:
            audio_path = data.get("file") or data.get("audio_url")
        if not audio_path:
            missing.append({"reason": "no audio field", "item": it})
            stats["missing"] += 1
            continue
        basename = Path(audio_path).name
        # moramo stripati sve sta nije ime zvuka
        stripped = basename
        if "-" in basename:
            parts = basename.split("-", 1)
            if len(parts) > 1:
                stripped = parts[1]

        # uzimamo samo prvu notaciju, dalje cemo prilagoditi ako ih bude vise OVDJE MIJENJAJ AKO SE NADJE VISE LABELA
        label_val = None
        annotations = it.get("annotations") or []
        if annotations:
            res = annotations[0].get("result") or []
            if res:
                v = res[0].get("value") or {}
                labels = v.get("labels") or []
                if labels:
                    label_val = labels[0]

        category = map_label_to_category(label_val)

        # vise kandidata imena
        candidates = [basename, stripped]
        candidates += [c[:-4] if c.lower().endswith('.wav') else c + '.wav' for c in list(candidates)]
        seen = set()
        candidates_unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                candidates_unique.append(c)

        found = None
        matched_candidate = None
        for cand in candidates_unique:
            found = find_file_by_basename(src_root, cand)
            if found:
                matched_candidate = cand
                break

        if not found:
            missing.append({"basename": basename, "tried": candidates_unique, "category": category})
            stats["missing"] += 1
            continue

        target_dir = dest_root / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / basename

        if args.apply:
            shutil.copy2(found, target_path)
            stats["copied"] += 1
            actions.append({"from": str(found), "to": str(target_path), "category": category})
        else:
            actions.append({"would_copy_from": str(found), "would_copy_to": str(target_path), "category": category})

    print("Summary:")
    print(f"  total items: {stats['total']}")
    print(f"  copied: {stats['copied']}")
    print(f"  missing: {stats['missing']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
