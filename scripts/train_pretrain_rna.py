from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig, OptimizerConfig
from perturb_jepa.training.checkpoint import save_checkpoint
from perturb_jepa.training.real_data import (
    load_yaml_or_json_config,
    make_token_mask,
    override_bridge_config_for_real_data,
    prepare_expression_matrix,
    raw_get,
    read_h5ad_subset,
)
from perturb_jepa.training.seed import seed_everything
from scripts.train_bridge import _run_synthetic_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain the RNA encoder scaffold.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated RNA/image synthetic batches.")
    parser.add_argument("--rna-anndata", type=Path, default=None, help="AnnData .h5ad file for real RNA pretraining.")
    parser.add_argument("--max-cells", type=int, default=None, help="Optional row subsample for Colab-scale runs.")
    parser.add_argument("--n-top-genes", type=int, default=None, help="Number of high-variance genes to tokenize.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config, raw_config = load_yaml_or_json_config(args.config)
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    rna_path = args.rna_anndata or _optional_path(raw_get(raw_config, ("data", "rna_anndata")))
    if args.synthetic or rna_path is None:
        if not args.synthetic:
            print("No RNA AnnData path provided; running synthetic pretrain. Pass --rna-anndata for real data.")
        _run_synthetic_training(config, stage="pretrain_rna", checkpoint_out=args.checkpoint_out)
    else:
        _run_real_rna_pretraining(
            config,
            rna_path=rna_path,
            max_cells=args.max_cells or raw_get(raw_config, ("data", "max_cells")),
            n_top_genes=args.n_top_genes or raw_get(raw_config, ("data", "n_top_genes")),
            batch_size=args.batch_size or raw_get(raw_config, ("training", "batch_size"), 32),
            mask_prob=args.mask_prob,
            normalize=config.data.rna_normalize,
            checkpoint_out=args.checkpoint_out,
        )
    return 0


def _run_real_rna_pretraining(
    config: ExperimentConfig,
    *,
    rna_path: Path,
    max_cells: int | None,
    n_top_genes: int | None,
    batch_size: int,
    mask_prob: float,
    normalize: bool,
    checkpoint_out: Path | None,
) -> None:
    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    adata = read_h5ad_subset(rna_path, max_cells=max_cells, seed=config.training.seed)
    expression, gene_token_ids = prepare_expression_matrix(
        adata.X,
        n_top_genes=n_top_genes,
        max_genes=config.model.rna.max_genes,
        normalize=normalize,
    )
    config = override_bridge_config_for_real_data(
        config,
        num_genes=expression.shape[1],
        max_genes=expression.shape[1],
    )
    model = config.build_model().to(device)
    optimizer = config.build_optimizer(model.rna_encoder.parameters())
    gene_ids = torch.as_tensor(np.broadcast_to(gene_token_ids[None, :], expression.shape).copy(), dtype=torch.long)
    values = torch.as_tensor(expression, dtype=torch.float32)
    loader = DataLoader(TensorDataset(gene_ids, values), batch_size=int(batch_size), shuffle=True)
    global_step = 0
    model.train()
    for epoch in range(max(1, config.training.steps)):
        for batch_gene_ids, batch_values in loader:
            batch_gene_ids = batch_gene_ids.to(device)
            batch_values = batch_values.to(device)
            token_mask = make_token_mask(tuple(batch_values.shape), mask_prob, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model.rna_encoder(batch_gene_ids, batch_values, token_mask=token_mask)
            target_mask = token_mask if token_mask.any() else torch.ones_like(token_mask)
            loss = torch.nn.functional.mse_loss(output.reconstruction[target_mask], batch_values[target_mask])
            loss.backward()
            if config.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.rna_encoder.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            if config.training.log_every > 0 and global_step % config.training.log_every == 0:
                print(f"stage=pretrain_rna step={global_step} rna_mask={float(loss.detach().cpu()):.4f}")
            global_step += 1
            if global_step >= config.training.steps:
                break
        if global_step >= config.training.steps:
            break
    if checkpoint_out is not None:
        save_checkpoint(
            checkpoint_out,
            model=model,
            optimizer=optimizer,
            trainer_state={"global_step": global_step, "genes": int(expression.shape[1])},
            experiment_config=config,
            metadata={"stage": "pretrain_rna", "rna_anndata": str(rna_path)},
        )


def _optional_path(value: object) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
