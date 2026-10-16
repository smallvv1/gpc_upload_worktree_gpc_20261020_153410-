#!/usr/local/bin/python3
"""Integrity and exact RGB-key matching for official DIFFER text resources."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/home/data")
TEXT_ROOT = DATA_ROOT / "DIFFER_TextCaption"
FACTOR_NAMES = ("Biometrics", "Hairstyle", "Clothing", "Pose", "Interaction", "Environment")


def _rgb_keys(dataset: str, split: str) -> list[str]:
    if dataset == "LTCC":
        return [path.stem for path in sorted((DATA_ROOT / "LTCC_ReID" / split).glob("*.png"))]
    root = DATA_ROOT / "prcc/rgb" / split
    return [path.relative_to(root).with_suffix("").as_posix() for path in sorted(root.glob("*/*.jpg"))]


def _match_report(rgb_keys: list[str], metadata: list[str]) -> dict[str, int]:
    rgb_counter = Counter(rgb_keys)
    text_counter = Counter(metadata)
    rgb_set = set(rgb_counter)
    text_set = set(text_counter)
    duplicate_occurrences = sum(count - 1 for count in text_counter.values() if count > 1)
    return {
        "RGB_TOTAL": len(rgb_keys),
        "TEXT_METADATA_TOTAL": len(metadata),
        "MATCHED": len(rgb_set & text_set),
        "MISSING": len(rgb_set - text_set),
        "EXTRA": len(text_set - rgb_set),
        "DUPLICATE_METADATA": duplicate_occurrences,
    }


def check_npz(dataset: str, split: str, path: Path, expected_count: int, expected_dim: int) -> None:
    print(f"\n== {dataset} {split}: {path} ==")
    with np.load(path, allow_pickle=True) as archive:
        print("keys:", archive.files)
        assert set(archive.files) == {"data", "metadata"}, archive.files
        data = archive["data"]
        metadata_array = archive["metadata"]
        metadata = [str(item) for item in metadata_array.tolist()]
        print("metadata.shape:", metadata_array.shape)
        print("data.shape:", data.shape)
        print("dtype:", data.dtype)
        print("min/max:", float(data.min()), float(data.max()))
        print("NaN count:", int(np.isnan(data).sum()))
        print("Inf count:", int(np.isinf(data).sum()))
        print("metadata first 10:", metadata[:10])
        assert metadata_array.shape == (expected_count,), metadata_array.shape
        assert data.ndim == 2 and data.shape[1] == expected_dim, data.shape
        assert data.shape[0] % metadata_array.shape[0] == 0, data.shape
        factor_count = data.shape[0] // metadata_array.shape[0]
        print("factor_count:", factor_count)
        print("factor mapping:", dict(enumerate(FACTOR_NAMES[:factor_count])))
        print("training selection:", {0: FACTOR_NAMES[0], 1: FACTOR_NAMES[1], 2: FACTOR_NAMES[2]} if dataset == "LTCC" else {0: FACTOR_NAMES[0], 1: FACTOR_NAMES[1]})
        assert factor_count == len(FACTOR_NAMES), factor_count
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

    report = _match_report(_rgb_keys(dataset, split), metadata)
    for key, value in report.items():
        print(f"{key}: {value}")
    assert report["RGB_TOTAL"] == expected_count
    assert report["TEXT_METADATA_TOTAL"] == expected_count
    assert report["MATCHED"] == expected_count
    assert report["MISSING"] == 0
    assert report["EXTRA"] == 0
    assert report["DUPLICATE_METADATA"] == 0
    print("FULL MATCH: PASS")


def check_summary(dataset: str, path: Path, expected_ids: int, feature_name: str, expected_dim: int) -> None:
    print(f"\n== {dataset} summary: {path} ==")
    with path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    assert isinstance(summary, dict)
    assert len(summary) == expected_ids, len(summary)
    arrays = []
    for pid, record in summary.items():
        assert "text" in record, pid
        assert feature_name in record, (pid, feature_name)
        array = np.asarray(record[feature_name])
        assert array.shape == (1, expected_dim), (pid, array.shape)
        arrays.append(array)
    features = np.concatenate(arrays, axis=0)
    print("ID records:", len(summary))
    print("feature key:", feature_name)
    print("features.shape/dtype:", features.shape, features.dtype)
    print("min/max:", float(features.min()), float(features.max()))
    print("NaN/Inf:", int(np.isnan(features).sum()), int(np.isinf(features).sum()))
    print("first keys:", list(summary)[:10])
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()
    print("SUMMARY: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "ltcc", "prcc"), default="all")
    args = parser.parse_args()
    if args.dataset in ("all", "ltcc"):
        check_npz(
            "LTCC", "train",
            TEXT_ROOT / "LTCC_ReID/EVA02-CLIP-B-16/train.npz",
            expected_count=9576, expected_dim=512,
        )
        check_summary(
            "LTCC", TEXT_ROOT / "LTCC_ReID/train_caption_summary_biometric.json",
            expected_ids=77, feature_name="ft_EVA02-CLIP-B-16", expected_dim=512,
        )
    if args.dataset in ("all", "prcc"):
        check_npz(
            "PRCC", "train",
            TEXT_ROOT / "PRCC/EVA02-CLIP-L-14/train.npz",
            expected_count=17896, expected_dim=768,
        )
        check_npz(
            "PRCC", "val",
            TEXT_ROOT / "PRCC/EVA02-CLIP-L-14/val.npz",
            expected_count=5002, expected_dim=768,
        )
        check_summary(
            "PRCC", TEXT_ROOT / "PRCC/train_caption_summary_biometric.json",
            expected_ids=150, feature_name="ft_EVA02-CLIP-L-14", expected_dim=768,
        )


if __name__ == "__main__":
    main()
