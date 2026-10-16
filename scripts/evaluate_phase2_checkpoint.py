"""Evaluate one Phase-2 checkpoint through the released evaluator.

This is an evaluation-only adapter for the current LTCC dataloader return
signature.  It does not alter model, loss, or metric implementations.
Run under torch.distributed.launch (one or more ranks); this adapter uses one
rank for an independent checkpoint check.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from config import cfg
from data import build_dataloader
from model import build_model
from processor import do_inference
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="DIFFER Phase-2 checkpoint evaluation")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("DIFFER", output_dir, if_train=False)
    logger.info("Phase-2 evaluation adapter args: %s", args)
    logger.info("Evaluating checkpoint: %s", cfg.TEST.WEIGHT)

    if cfg.DATA.DATASET != "ltcc":
        raise ValueError("This adapter is currently scoped to LTCC")

    # Current official LTCC build_dataloader returns seven values.
    _, _, _, dataset, _, val_loader, _ = build_dataloader(cfg)
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    state_dict = torch.load(cfg.TEST.WEIGHT, map_location="cpu")
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Checkpoint load missing_keys=%s unexpected_keys=%s", missing, unexpected)

    do_inference(cfg, model, dataset, val_loader=val_loader)


if __name__ == "__main__":
    main()
