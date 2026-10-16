#!/usr/local/bin/python3
"""Paper-faithful DIFFER PHASE 1 regression checks (no formal training)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/data")
sys.path.insert(0, str(REPO_ROOT))


class _LazyFeatures:
    """Shape-correct caption fixture without allocating the official feature arrays."""

    def __init__(self, metadata_count: int, factor_count: int, dim: int):
        self.shape = (metadata_count * factor_count, dim)
        self.dtype = np.dtype(np.float32)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError(f"fixture expects a slice, got {key!r}")
        start, stop, step = key.indices(self.shape[0])
        return np.zeros((len(range(start, stop, step)), self.shape[1]), dtype=self.dtype)


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_caption_npz(metadata: list[str], factor_count: int, dim: int):
    return {
        "metadata": np.asarray(metadata, dtype=object),
        "data": _LazyFeatures(len(metadata), factor_count, dim),
    }


def check_datasets() -> None:
    ltcc_module = _load_source_module("phase1_ltcc", REPO_ROOT / "data/datasets/ltcc.py")
    prcc_module = _load_source_module("phase1_prcc", REPO_ROOT / "data/datasets/prcc.py")

    ltcc_metadata = [path.stem for path in sorted((DATA_ROOT / "LTCC_ReID/train").glob("*.png"))]
    with mock.patch.object(
        ltcc_module.np,
        "load",
        return_value=_mock_caption_npz(ltcc_metadata, factor_count=3, dim=512),
    ):
        ltcc = ltcc_module.LTCC(
            root=str(DATA_ROOT),
            caption_dir="/phase1/mock/LTCC",
            caption_model="EVA02-CLIP-B-16",
            load_sum_ft=False,
            bio_index="0",
            nonbio_index=(1, 2),
        )
    ltcc_result = (
        ltcc.dataset_dir,
        len(ltcc.train),
        len(ltcc.query),
        len(ltcc.gallery),
        ltcc.num_train_pids,
        ltcc.num_train_clothes,
    )
    ltcc_expected = (str(DATA_ROOT / "LTCC_ReID"), 9576, 493, 7050, 77, 256)
    assert ltcc_result == ltcc_expected, (ltcc_result, ltcc_expected)
    print("LTCC loader: PASS")
    print(f"  dataset_dir: {ltcc.dataset_dir}")
    print(f"  train/query/gallery: {len(ltcc.train)}/{len(ltcc.query)}/{len(ltcc.gallery)}")
    print(f"  num_train_pids/num_clothes: {ltcc.num_train_pids}/{ltcc.num_train_clothes}")

    prcc_train_dir = DATA_ROOT / "prcc/rgb/train"
    prcc_val_dir = DATA_ROOT / "prcc/rgb/val"
    prcc_metadata = {
        "train.npz": [str(path.relative_to(prcc_train_dir).with_suffix("")) for path in sorted(prcc_train_dir.glob("*/*.jpg"))],
        "val.npz": [str(path.relative_to(prcc_val_dir).with_suffix("")) for path in sorted(prcc_val_dir.glob("*/*.jpg"))],
    }

    def prcc_np_load(path, *args, **kwargs):
        metadata = prcc_metadata[Path(path).name]
        return _mock_caption_npz(metadata, factor_count=2, dim=768)

    with mock.patch.object(prcc_module.np, "load", side_effect=prcc_np_load):
        prcc = prcc_module.PRCC(
            root=str(DATA_ROOT),
            caption_dir="/phase1/mock/PRCC",
            caption_model="EVA02-CLIP-L-14",
            load_sum_ft=False,
            bio_index="0",
            nonbio_index=(1,),
        )
    prcc_result = (
        prcc.dataset_dir,
        len(prcc.train),
        len(prcc.query_same),
        len(prcc.query_diff),
        len(prcc.gallery),
        prcc.num_train_pids,
        prcc.num_train_clothes,
    )
    prcc_expected = (str(DATA_ROOT / "prcc"), 17896, 3873, 3543, 3384, 150, 300)
    assert prcc_result == prcc_expected, (prcc_result, prcc_expected)
    print("PRCC loader: PASS")
    print(f"  dataset_dir: {prcc.dataset_dir}")
    print(f"  train/query_same/query_diff/gallery: {len(prcc.train)}/{len(prcc.query_same)}/{len(prcc.query_diff)}/{len(prcc.gallery)}")
    print(f"  num_train_pids/num_clothes: {prcc.num_train_pids}/{prcc.num_train_clothes}")


def check_eva_shapes() -> None:
    from model.eva_meta_bio import Eva

    common = dict(
        img_size=28,
        patch_size=14,
        num_classes=77,
        embed_dim=1024,
        clip_feat_dim=512,
        last_layer="clipFc",
        global_pool="token",
        depth=1,
        num_heads=16,
        camera_xishu=1,
        camera_num=2,
        add_nonBio=False,
        nonBio_num=2,
        use_rot_pos_emb=False,
    )
    images = torch.randn(2, 3, 28, 28)
    cameras = torch.zeros(2, dtype=torch.long)
    for variant, reverse_bio in (("Baseline", False), ("Bio-only", False), ("Bio+nonbio/Full DIFFER", True)):
        model = Eva(reverse_bio=reverse_bio, **common)
        model.eval()
        with torch.no_grad():
            global_feature = model(images, cameras)
            classifier_output = model.head(global_feature)
        assert global_feature.shape == (2, 1024), global_feature.shape
        assert classifier_output.shape == (2, 77), classifier_output.shape
        print(f"{variant} EVA global/classifier: PASS {tuple(global_feature.shape)} -> {tuple(classifier_output.shape)}")


def check_nonbio_shapes() -> None:
    batch, factors, dim = 8, 2, 512
    image_feature_reverse = torch.randn(batch, factors, dim)
    text_features_nonbio = torch.randn(batch, factors, dim)
    for index in range(factors):
        image_feature_reverse_i = torch.nn.functional.normalize(
            image_feature_reverse[:, index], dim=-1
        )
        logits = image_feature_reverse_i @ text_features_nonbio[:, index].T
        labels = torch.arange(
            image_feature_reverse_i.shape[0], device=image_feature_reverse_i.device
        )
        assert image_feature_reverse_i.shape == (8, 512)
        assert logits.shape == (8, 8)
        assert labels.shape == (8,)
        print(f"nonbio factor {index}: PASS feature={tuple(image_feature_reverse_i.shape)} logits={tuple(logits.shape)} labels={tuple(labels.shape)}")


def _yacs_value(config_path: Path):
    from config import cfg

    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_path))
    return list(local_cfg.DATA.NOBIO_INDEX)


def check_config() -> None:
    ltcc = _yacs_value(REPO_ROOT / "configs/ltcc/eva02_l_bio.yml")
    prcc = _yacs_value(REPO_ROOT / "configs/prcc/eva02_l_bio.yml")
    assert ltcc == [1, 2], ltcc
    assert prcc == [1], prcc
    print(f"NOBIO_INDEX LTCC: PASS {ltcc}")
    print(f"NOBIO_INDEX PRCC: PASS {prcc}")

    for relative in ("configs/ltcc/eva02_l_bio.yml", "configs/prcc/eva02_l_bio.yml"):
        with (REPO_ROOT / relative).open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)["MODEL"]["LAST_LAYER"]
        assert value == "clipFc", (relative, value)
        print(f"LAST_LAYER {relative}: PASS {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        choices=("all", "datasets", "eva", "nonbio", "config"),
        default="all",
    )
    args = parser.parse_args()
    checks = {
        "datasets": check_datasets,
        "eva": check_eva_shapes,
        "nonbio": check_nonbio_shapes,
        "config": check_config,
    }
    selected = checks.items() if args.check == "all" else ((args.check, checks[args.check]),)
    for name, check in selected:
        print(f"\n== {name} ==")
        check()


if __name__ == "__main__":
    main()
