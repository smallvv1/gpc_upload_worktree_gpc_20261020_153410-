#!/usr/local/bin/python3
"""Run exactly 20 updates through processor.do_train with diagnostic observation."""

from __future__ import annotations

import gc
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter as RealSummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import processor.processor as processor_module
from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from data.samplers import DistributedRandomIdentitySampler
from model import build_model
from scripts.localize_amp_overflow_phase1 import GROUPS, group_stats, set_seed
from scripts.qualify_amp_initial_scale_phase1 import make_dataset_config


class RegressionComplete(RuntimeError):
    pass


class LimitedLoader:
    def __init__(self, loader, limit, device):
        self.loader = loader
        self.limit = limit
        self.device = device
        self.batch_size = loader.batch_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.limit):
            batch = next(iterator)
            yield {
                key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
        raise RegressionComplete(f"completed {self.limit} processor iterations")


class CaptureWriter:
    instances = []

    def __init__(self, log_dir):
        self.real = RealSummaryWriter(log_dir=log_dir)
        self.scalars = {}
        CaptureWriter.instances.append(self)

    def add_scalar(self, name, value, step):
        numeric = float(value)
        self.scalars.setdefault(int(step), {})[name] = numeric
        self.real.add_scalar(name, numeric, step)

    def close(self):
        self.real.flush()
        self.real.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ltcc", "prcc"), default="ltcc")
    args = parser.parse_args()
    local_cfg = make_dataset_config(args.dataset)
    set_seed(local_cfg.SOLVER.SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dataset = build_dataset(local_cfg)
    train_transform, _ = build_img_transforms(local_cfg)
    sampler = DistributedRandomIdentitySampler(
        dataset.train,
        num_instances=local_cfg.DATA.NUM_INSTANCES,
        num_replicas=1,
        rank=0,
        seed=local_cfg.SOLVER.SEED,
    )
    loader = DataLoader(
        ImageDataset(dataset.train, transform=train_transform),
        sampler=sampler,
        batch_size=local_cfg.DATA.BATCH_SIZE,
        num_workers=local_cfg.DATA.NUM_WORKERS,
        pin_memory=local_cfg.DATA.PIN_MEMORY,
        drop_last=True,
        generator=torch.Generator().manual_seed(local_cfg.SOLVER.SEED),
    )
    # Official DataLoaderX performs this device transfer during preload.  The
    # finite wrapper reproduces that behavior while limiting processor.do_train.
    limited_loader = LimitedLoader(loader, 20, device)
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    model.train()

    OriginalGradScaler = processor_module.amp.GradScaler

    class AuditGradScaler(OriginalGradScaler):
        records = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pending_loss = None
            self.pending = None
            print("PROCESSOR_GRADSCALER_INIT", json.dumps({
                "args": args,
                "kwargs": kwargs,
                "actual_scale": float(self.get_scale()),
            }, sort_keys=True))

        def scale(self, outputs):
            self.pending_loss = {
                "value": float(outputs.detach()),
                "dtype": str(outputs.dtype),
                "finite": bool(torch.isfinite(outputs)),
            }
            return super().scale(outputs)

        def step(self, optimizer, *args, **kwargs):
            scale_before = float(self.get_scale())
            monitored = {
                "backbone": model.patch_embed.proj.weight,
                "classifier": model.head.weight,
            }
            before = {name: parameter.detach().clone() for name, parameter in monitored.items()}
            result = super().step(optimizer, *args, **kwargs)
            found_inf_state = self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
            found_inf = sum(float(value.item()) for value in found_inf_state.values())
            deltas = {
                name: float((parameter.detach() - before[name]).float().abs().max())
                for name, parameter in monitored.items()
            }
            groups = {group: group_stats(model, prefixes) for group, prefixes in GROUPS.items()}
            self.pending = {
                "iteration": len(self.records) + 1,
                "scale_before": scale_before,
                "found_inf": found_inf,
                "step_skipped": found_inf > 0,
                "parameter_updated": all(delta > 0 for delta in deltas.values()),
                "parameter_deltas": deltas,
                "groups": groups,
                "loss": self.pending_loss,
            }
            return result

        def update(self, *args, **kwargs):
            result = super().update(*args, **kwargs)
            self.pending["scale_after"] = float(self.get_scale())
            self.records.append(self.pending)
            if len(self.records) == 1 or len(self.records) % 5 == 0:
                print("PROCESSOR_REGRESSION_PROGRESS", json.dumps(self.pending, sort_keys=True))
            self.pending = None
            return result

    diagnostics_dir = REPO_ROOT / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    selected_scale = 512 if args.dataset == "prcc" else 1024
    output_dir = diagnostics_dir / f"processor_{args.dataset}_scale{selected_scale}_regression"
    local_cfg = local_cfg.clone()
    local_cfg.defrost()
    local_cfg.OUTPUT_DIR = str(output_dir)
    local_cfg.freeze()

    processor_module.amp.GradScaler = AuditGradScaler
    processor_module.SummaryWriter = CaptureWriter
    try:
        processor_module.do_train(local_cfg, model, limited_loader, 0, dataset)
    except RegressionComplete as error:
        print("PROCESSOR_LIMIT_REACHED:", error)
    finally:
        processor_module.amp.GradScaler = OriginalGradScaler
        processor_module.SummaryWriter = RealSummaryWriter
        for writer in CaptureWriter.instances:
            writer.close()

    records = AuditGradScaler.records
    scalar_records = {}
    for writer in CaptureWriter.instances:
        scalar_records.update(writer.scalars)
    observed_losses = [
        value for values in scalar_records.values()
        for name, value in values.items() if name.startswith("loss/")
    ]
    nonfinite_scalar_count = sum(not math.isfinite(value) for value in observed_losses)
    nonfinite_gradient_iterations = sum(
        any(stats["nan_count"] or stats["inf_count"] for stats in record["groups"].values())
        for record in records
    )
    skipped = sum(record["step_skipped"] for record in records)
    updated = sum(record["parameter_updated"] for record in records)
    summary = {
        "iterations": len(records),
        "initial_scale": records[0]["scale_before"] if records else None,
        "final_scale": records[-1]["scale_after"] if records else None,
        "nonfinite_logged_losses": nonfinite_scalar_count,
        "nonfinite_gradient_iterations": nonfinite_gradient_iterations,
        "skipped_steps": skipped,
        "successful_parameter_updates": updated,
        "status": "PASS" if (
            len(records) == 20
            and nonfinite_scalar_count == 0
            and nonfinite_gradient_iterations == 0
            and skipped == 0
            and updated == 20
        ) else "FAIL",
    }
    print("PROCESSOR_REGRESSION_SUMMARY", json.dumps(summary, sort_keys=True))
    (diagnostics_dir / f"processor_{args.dataset}_scale{selected_scale}_20iter.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, sort_keys=True) + "\n"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
