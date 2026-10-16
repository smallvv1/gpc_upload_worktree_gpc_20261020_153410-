#!/usr/local/bin/python3
"""Official-config DIFFER AMP trajectory preflight (no formal epoch training)."""

from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data import build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from data.samplers import DistributedRandomIdentitySampler
from loss import make_loss
from loss.triplet_loss import TripletLoss
from model import build_model
from model.eva_meta_bio import GradientReversalLayer
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


class ScalarCapture:
    def __init__(self):
        self.values: dict[str, float] = {}

    def add_scalar(self, name, value, step):
        self.values[name] = float(value)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def module_grad_stats(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, float | int | bool | None]:
    tensors = 0
    nan_count = 0
    inf_count = 0
    finite_count = 0
    max_abs_finite: float | None = None
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes) and parameter.grad is not None:
            gradient = parameter.grad.detach().double()
            nan_count += int(torch.isnan(gradient).sum().item())
            inf_count += int(torch.isinf(gradient).sum().item())
            finite = gradient[torch.isfinite(gradient)]
            finite_count += finite.numel()
            if finite.numel():
                tensor_max = float(finite.abs().max().item())
                max_abs_finite = tensor_max if max_abs_finite is None else max(max_abs_finite, tensor_max)
            tensors += 1
    return {
        "grad_exists": tensors > 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_count": finite_count,
        "max_abs_finite_grad": max_abs_finite,
        "tensor_count": tensors,
    }


def grl_direction_check(device: torch.device) -> None:
    direct = torch.tensor([1.0, -2.0, 3.0], device=device, requires_grad=True)
    direct.sum().backward()
    direct_grad = direct.grad.detach().clone()
    reversed_input = torch.tensor([1.0, -2.0, 3.0], device=device, requires_grad=True)
    GradientReversalLayer(alpha=1.0)(reversed_input).sum().backward()
    reversed_grad = reversed_input.grad.detach().clone()
    passed = torch.equal(reversed_grad, -direct_grad)
    print("GRL direct gradient:", direct_grad.cpu().tolist())
    print("GRL reversed gradient:", reversed_grad.cpu().tolist())
    print("GRL_DIRECTION_PASS:", passed)
    assert passed


def make_config(dataset_name: str):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(REPO_ROOT / f"configs/{dataset_name}/eva02_l_bio.yml"))
    local_cfg.defrost()
    local_cfg.MODEL.LOSS_TYPE = ["ce", "triplet", "clipBio", "clipBioReverse"]
    local_cfg.DATA.ROOT = "/home/data"
    if dataset_name == "ltcc":
        local_cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
        local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-B-16"
    else:
        local_cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/PRCC"
        local_cfg.DATA.TEXT_MODEL_VERSION = "EVA02-CLIP-L-14"
    local_cfg.freeze()
    return local_cfg


def one_batch(dataset_name: str) -> None:
    local_cfg = make_config(dataset_name)
    set_seed(local_cfg.SOLVER.SEED)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("dataset:", dataset_name)
    print("device:", torch.cuda.get_device_name(0))
    print("batch/image size:", local_cfg.DATA.BATCH_SIZE, (local_cfg.DATA.IMG_HEIGHT, local_cfg.DATA.IMG_WIDTH))
    print("loss types:", list(local_cfg.MODEL.LOSS_TYPE))
    print("NOBIO_INDEX:", list(local_cfg.DATA.NOBIO_INDEX))

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
    model = build_model(local_cfg, dataset.num_train_pids, dataset.num_camera).to(device)
    model.train()
    loss_fn, center_criterion = make_loss(local_cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(local_cfg, model, center_criterion)
    scheduler = create_scheduler(local_cfg, optimizer)
    scheduler.step(local_cfg.TRAIN.START_EPOCH)
    scaler = amp.GradScaler()
    print("scheduler stepped at epoch start: True")
    print("scheduler epoch argument:", local_cfg.TRAIN.START_EPOCH)
    print("optimizer LR after scheduler step:", sorted({group["lr"] for group in optimizer.param_groups}))
    print("GradScaler initial scale:", scaler.get_scale())

    grl_records: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def grl_hook(module, grad_input, grad_output):
        if grad_input and grad_input[0] is not None and grad_output and grad_output[0] is not None:
            grl_records.append((grad_input[0].detach(), grad_output[0].detach(), module.alpha))

    hook_handles = [
        reverse_head.grl.register_full_backward_hook(grl_hook)
        for reverse_head in model.head_clip_bioReverse
    ]
    gradient_groups = {
        "EVA backbone": ("patch_embed.", "blocks."),
        "ID classifier": ("head.",),
        "biometric transformer": ("head_bio.",),
        "bio projector": ("head_clip_bio.",),
        "nonbio reverse projector": ("head_clip_bioReverse.",),
    }
    selected_parameters = {
        "backbone.patch_embed.proj.weight": model.patch_embed.proj.weight,
        "classifier.head.weight": model.head.weight,
    }
    scale_history = []
    skipped_history = []
    update_history = []
    trajectory = []
    consecutive_stable = 0
    first_stable_scale = None
    optimizer_update_count = 0
    scheduler_step_count = 1
    loader_iterator = iter(loader)

    for iteration in range(1, 11):
        print(f"\n===== ITERATION {iteration} =====")
        batch = next(loader_iterator)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        writer = ScalarCapture()
        grl_records.clear()
        scale_before = float(scaler.get_scale())
        images = batch["image"]
        targets = batch["pid"]
        cameras = batch["camid"]
        clothes = batch["clothes_id"]
        captions = batch["caption_feature"]
        with amp.autocast(enabled=True):
            score, feat = model(images, cam_label=cameras)
        global_feature = feat[0]
        total_loss = loss_fn(
            score, feat, targets, cameras, captions,
            clothes_ids=clothes, train_writer=writer, step=iteration,
        )

        bio_projected = score["clip_bio_score"]
        if bio_projected.ndim == 2:
            bio_projected = bio_projected.unsqueeze(1)
        nonbio_projected = score["clip_bio_reverse_score"]
        text_bio = captions[:, : bio_projected.shape[1]]
        text_nonbio = captions[:, bio_projected.shape[1] :]
        bio_logits = torch.stack([
            F.normalize(bio_projected[:, index], dim=-1)
            @ F.normalize(text_bio[:, index].type_as(bio_projected), dim=-1).T
            for index in range(bio_projected.shape[1])
        ], dim=1)
        nonbio_logits = torch.stack([
            F.normalize(nonbio_projected[:, index], dim=-1)
            @ F.normalize(text_nonbio[:, index].type_as(nonbio_projected), dim=-1).T
            for index in range(nonbio_projected.shape[1])
        ], dim=1)
        if iteration == 1:
            print("images:", tuple(images.shape))
            print("global image feature:", tuple(global_feature.shape))
            print("projected biometric feature:", tuple(bio_projected.shape))
            print("projected non-biometric feature:", tuple(nonbio_projected.shape))
            print("text bio:", tuple(text_bio.shape))
            print("text nonbio:", tuple(text_nonbio.shape))
            print("ID logits:", tuple(score["cls_score"].shape))
            print("Bio CLIP logits:", tuple(bio_logits.shape))
            print("Nonbio reverse logits:", tuple(nonbio_logits.shape))

        labels = torch.arange(images.shape[0], device=device)
        detached_losses = {
            "id": F.cross_entropy(score["cls_score"].detach(), targets),
            "triplet": TripletLoss()(global_feature.detach(), targets)[0],
        }
        for index in range(bio_logits.shape[1]):
            detached_losses[f"clip_bio_{index}"] = F.cross_entropy(bio_logits[:, index].detach(), labels)
        for index in range(nonbio_logits.shape[1]):
            detached_losses[f"clip_bio_reverse_{index}"] = F.cross_entropy(nonbio_logits[:, index].detach(), labels)
        for name, value in detached_losses.items():
            print(f"loss {name}: value={float(value)} dtype={value.dtype} finite={bool(torch.isfinite(value))}")
            assert bool(torch.isfinite(value))
        print(f"total_loss: value={float(total_loss.detach())} dtype={total_loss.dtype} finite={bool(torch.isfinite(total_loss))}")
        print("official writer scalars:", {key: value for key, value in writer.values.items() if key.startswith("loss/")})
        assert total_loss.ndim == 0 and bool(torch.isfinite(total_loss))
        assert abs(float(total_loss.detach()) - sum(float(value) for value in detached_losses.values())) < 0.05

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        all_critical_finite = True
        for name, prefixes in gradient_groups.items():
            stats = module_grad_stats(model, prefixes)
            print(f"gradient {name}:", stats)
            finite = stats["grad_exists"] and stats["nan_count"] == 0 and stats["inf_count"] == 0
            all_critical_finite = all_critical_finite and finite

        grl_path_pass = len(grl_records) == len(model.head_clip_bioReverse) and all(
            torch.allclose(gradient_input, -alpha * gradient_output, equal_nan=True)
            for gradient_input, gradient_output, alpha in grl_records
        )
        print("actual GRL hooks observed:", len(grl_records))
        print("ACTUAL_GRL_DIRECTION_PASS:", grl_path_pass)
        assert grl_path_pass
        found_inf_state = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        found_inf = sum(float(value.item()) for value in found_inf_state.values())
        parameters_before = {name: parameter.detach().clone() for name, parameter in selected_parameters.items()}
        scaler.step(optimizer)
        parameter_deltas = {
            name: float((parameter.detach() - parameters_before[name]).float().abs().max())
            for name, parameter in selected_parameters.items()
        }
        parameter_updated = any(delta > 0 for delta in parameter_deltas.values())
        step_skipped = found_inf > 0
        if parameter_updated:
            optimizer_update_count += 1
        scaler.update()
        scale_after = float(scaler.get_scale())
        current_lr = sorted({float(group["lr"]) for group in optimizer.param_groups})
        print("scale_before:", scale_before)
        print("scale_after:", scale_after)
        print("found_inf:", found_inf)
        print("STEP_SKIPPED:", step_skipped)
        print("parameter max_abs_delta:", parameter_deltas)
        print("PARAMETER_UPDATED:", parameter_updated)
        print("scheduler this iteration: NOT STEPPED (epoch-level scheduler already stepped before iteration 1)")
        print("optimizer update count:", optimizer_update_count)
        print("scheduler step count:", scheduler_step_count)
        print("current LR:", current_lr)
        scale_history.append((scale_before, scale_after))
        skipped_history.append(step_skipped)
        update_history.append(parameter_updated)
        trajectory.append({
            "iteration": iteration,
            "scale_before": scale_before,
            "scale_after": scale_after,
            "found_inf": found_inf,
            "step_skipped": step_skipped,
            "parameter_updated": parameter_updated,
            "loss": float(total_loss.detach()),
            "loss_finite": bool(torch.isfinite(total_loss)),
            "lr": current_lr,
            "all_critical_finite": all_critical_finite,
        })
        stable = (
            found_inf == 0
            and not step_skipped
            and parameter_updated
            and all_critical_finite
            and scale_after == scale_before
        )
        consecutive_stable = consecutive_stable + 1 if stable else 0
        if consecutive_stable >= 3:
            first_stable_scale = scale_after
            print("THREE_CONSECUTIVE_STABLE_ITERATIONS: True")
            break

    for handle in hook_handles:
        handle.remove()
    print("\nScale trajectory:", scale_history)
    print("Skipped history:", skipped_history)
    print("Parameter update history:", update_history)
    print("Trajectory records:", trajectory)
    first_successful = next((index + 1 for index, updated in enumerate(update_history) if updated), None)
    print("First successful optimizer update:", first_successful)
    print("FIRST_STABLE_SCALE:", first_stable_scale)
    print("SKIPPED_STEPS_BEFORE_STABLE:", sum(skipped_history))
    print("SCHEDULER_DESYNC:", any(skipped_history) and scheduler_step_count > 0)
    status = "STABLE AFTER MULTIPLE BACKOFFS" if first_stable_scale is not None else "NOT STABLE BY ITERATION 10"
    print(f"{dataset_name.upper()} NATURAL SCALE DIAGNOSTIC: {status}")

    del model, optimizer, optimizer_center, center_criterion
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("ltcc", "prcc"))
    args = parser.parse_args()
    one_batch(args.dataset)


if __name__ == "__main__":
    main()
