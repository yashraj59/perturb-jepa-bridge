from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.data.conditions import MetadataVocab
from perturb_jepa.data.schema import make_condition_id, normalize_value
from perturb_jepa.models.counterfactual import (
    CounterfactualResponsePredictor,
    PerturbationConditionEncoder,
    counterfactual_mmd_loss,
    gaussian_nll_loss,
)
from perturb_jepa.training.real_data import (
    load_yaml_or_json_config,
    override_bridge_config_for_real_data,
    prepare_expression_matrix,
    raw_get,
    read_h5ad_subset,
)
from perturb_jepa.training.seed import seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Perturb-JEPA counterfactual heads/objectives.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated synthetic batches.")
    parser.add_argument("--rna-anndata", type=Path, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--control-values", default="control,ctrl,dmso,vehicle,untreated,mock")
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
            print("No RNA AnnData path provided; running synthetic counterfactual. Pass --rna-anndata for real data.")
        _run_synthetic_counterfactual(config, checkpoint_out=args.checkpoint_out)
    else:
        _run_real_rna_counterfactual(
            config,
            rna_path=rna_path,
            max_cells=args.max_cells or raw_get(raw_config, ("data", "max_cells")),
            n_top_genes=args.n_top_genes or raw_get(raw_config, ("data", "n_top_genes")),
            batch_size=args.batch_size or raw_get(raw_config, ("training", "batch_size"), 32),
            control_values=[value.strip() for value in args.control_values.split(",") if value.strip()],
            checkpoint_out=args.checkpoint_out,
        )
    return 0


def _run_synthetic_counterfactual(config: ExperimentConfig, *, checkpoint_out: Path | None) -> None:
    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    shared_dim = config.model.shared_dim
    prototypes = config.model.num_bag_prototypes
    condition_encoder = PerturbationConditionEncoder(
        num_conditions=config.synthetic.num_perturbations,
        feature_dim=2,
        hidden_dim=shared_dim,
        output_dim=shared_dim,
    ).to(device)
    predictor = CounterfactualResponsePredictor(
        prototype_dim=shared_dim,
        condition_dim=shared_dim,
        hidden_dim=shared_dim,
    ).to(device)
    optimizer = config.build_optimizer([*condition_encoder.parameters(), *predictor.parameters()])

    fixed_effects = torch.randn(config.synthetic.num_perturbations, shared_dim, device=device) * 0.2
    for step in range(config.training.steps):
        batch_size = config.synthetic.batch_size
        control = torch.randn(batch_size, prototypes, shared_dim, device=device)
        perturbation_id = torch.randint(0, config.synthetic.num_perturbations, (batch_size,), device=device)
        dose = torch.rand(batch_size, 1, device=device)
        time = torch.rand(batch_size, 1, device=device)
        condition_features = torch.cat((dose.log1p(), time), dim=-1)
        true_delta = fixed_effects[perturbation_id].unsqueeze(1) * (0.5 + dose.unsqueeze(-1))
        observed_treated = control + true_delta + 0.02 * torch.randn_like(control)

        optimizer.zero_grad(set_to_none=True)
        condition_embedding = condition_encoder(perturbation_id, condition_features)
        prediction = predictor(control, condition_embedding)
        nll = gaussian_nll_loss(prediction.treated_mu, prediction.treated_logvar, observed_treated)
        mmd = counterfactual_mmd_loss(prediction.treated_mu, observed_treated)
        total = nll + 0.1 * mmd
        total.backward()
        if config.training.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [*condition_encoder.parameters(), *predictor.parameters()],
                config.training.grad_clip_norm,
            )
        optimizer.step()
        if config.training.log_every > 0 and (step + 1) % config.training.log_every == 0:
            print(
                f"stage=counterfactual step={step} "
                f"counterfactual_mmd={float(mmd.detach().cpu()):.4f} "
                f"counterfactual_nll={float(nll.detach().cpu()):.4f} "
                f"total={float(total.detach().cpu()):.4f}"
            )

    if checkpoint_out is not None:
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "condition_encoder": condition_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "experiment_config": config.to_dict(),
                "trainer_state": {"global_step": config.training.steps},
            },
            checkpoint_out,
        )


def _run_real_rna_counterfactual(
    config: ExperimentConfig,
    *,
    rna_path: Path,
    max_cells: int | None,
    n_top_genes: int | None,
    batch_size: int,
    control_values: list[str],
    checkpoint_out: Path | None,
) -> None:
    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    adata = read_h5ad_subset(rna_path, max_cells=max_cells, seed=config.training.seed)
    expression, _ = prepare_expression_matrix(
        adata.X,
        n_top_genes=n_top_genes,
        max_genes=config.model.rna.max_genes,
    )
    metadata = pd.DataFrame(adata.obs).reset_index(drop=True)
    pairs, vocab = _control_treated_pairs(expression, metadata, control_values)
    if not pairs:
        raise ValueError(
            "No control-relative condition pairs found. Provide controls via --control-values "
            "and ensure controls share cell_line/time with treated conditions."
        )
    config = override_bridge_config_for_real_data(
        config,
        num_genes=expression.shape[1],
        max_genes=expression.shape[1],
        metadata_vocab=vocab,
    )
    condition_encoder = PerturbationConditionEncoder(
        num_perturbations=vocab.num_perturbations,
        num_cell_lines=vocab.num_cell_lines,
        hidden_dim=config.model.shared_dim,
        output_dim=config.model.shared_dim,
    ).to(device)
    predictor = CounterfactualResponsePredictor(
        prototype_dim=expression.shape[1],
        condition_dim=config.model.shared_dim,
        hidden_dim=max(config.model.shared_dim, expression.shape[1]),
    ).to(device)
    optimizer = config.build_optimizer([*condition_encoder.parameters(), *predictor.parameters()])
    tensors = TensorDataset(
        torch.as_tensor(np.stack([pair["control"] for pair in pairs]), dtype=torch.float32),
        torch.as_tensor(np.stack([pair["treated"] for pair in pairs]), dtype=torch.float32),
        torch.tensor([pair["perturbation_id"] for pair in pairs], dtype=torch.long),
        torch.tensor([pair["cell_line_id"] for pair in pairs], dtype=torch.long),
        torch.tensor([pair["dose"] for pair in pairs], dtype=torch.float32),
        torch.tensor([pair["time"] for pair in pairs], dtype=torch.float32),
    )
    loader = DataLoader(tensors, batch_size=int(batch_size), shuffle=True)
    global_step = 0
    for _ in range(max(1, config.training.steps)):
        for control, treated, perturbation_id, cell_line_id, dose, time in loader:
            control = control.to(device).unsqueeze(1)
            treated = treated.to(device).unsqueeze(1)
            perturbation_id = perturbation_id.to(device)
            cell_line_id = cell_line_id.to(device)
            dose = dose.to(device)
            time = time.to(device)
            optimizer.zero_grad(set_to_none=True)
            condition_embedding = condition_encoder(
                perturbation_id=perturbation_id,
                dose=dose,
                time=time,
                cell_line_id=cell_line_id,
            )
            prediction = predictor(control, condition_embedding)
            nll = gaussian_nll_loss(prediction.treated_mu, prediction.treated_logvar, treated)
            mmd = counterfactual_mmd_loss(prediction.treated_mu, treated)
            total = nll + 0.1 * mmd
            total.backward()
            if config.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [*condition_encoder.parameters(), *predictor.parameters()],
                    config.training.grad_clip_norm,
                )
            optimizer.step()
            if config.training.log_every > 0 and global_step % config.training.log_every == 0:
                print(
                    f"stage=counterfactual step={global_step} "
                    f"pairs={len(pairs)} counterfactual_mmd={float(mmd.detach().cpu()):.4f} "
                    f"counterfactual_nll={float(nll.detach().cpu()):.4f} total={float(total.detach().cpu()):.4f}"
                )
            global_step += 1
            if global_step >= config.training.steps:
                break
        if global_step >= config.training.steps:
            break
    if checkpoint_out is not None:
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "condition_encoder": condition_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "experiment_config": config.to_dict(),
                "trainer_state": {"global_step": global_step, "pairs": len(pairs)},
                "metadata": {"stage": "counterfactual", "rna_anndata": str(rna_path)},
            },
            checkpoint_out,
        )


def _control_treated_pairs(
    expression: np.ndarray,
    metadata: pd.DataFrame,
    control_values: list[str],
) -> tuple[list[dict[str, object]], MetadataVocab]:
    metadata = metadata.copy()
    metadata["condition_id"] = [make_condition_id(row) for row in metadata.to_dict(orient="records")]
    vocab = MetadataVocab.from_frame(metadata)
    control_tokens = {value.strip().lower() for value in control_values}
    metadata["_is_control"] = metadata["perturbation"].map(lambda value: normalize_value(value).lower() in control_tokens)
    prototypes: dict[str, np.ndarray] = {}
    rows_by_condition: dict[str, dict[str, object]] = {}
    for condition_id, group in metadata.groupby("condition_id", sort=True):
        indices = group.index.to_numpy()
        prototypes[condition_id] = expression[indices].mean(axis=0)
        rows_by_condition[condition_id] = group.iloc[0].to_dict()
    controls = {
        (normalize_value(row.get("cell_line")), normalize_value(row.get("time"))): prototypes[condition_id]
        for condition_id, row in rows_by_condition.items()
        if bool(row.get("_is_control"))
    }
    fallback_controls = {
        normalize_value(row.get("cell_line")): prototypes[condition_id]
        for condition_id, row in rows_by_condition.items()
        if bool(row.get("_is_control"))
    }
    pairs: list[dict[str, object]] = []
    for condition_id, row in rows_by_condition.items():
        if bool(row.get("_is_control")):
            continue
        control = controls.get((normalize_value(row.get("cell_line")), normalize_value(row.get("time"))))
        if control is None:
            control = fallback_controls.get(normalize_value(row.get("cell_line")))
        if control is None:
            continue
        encoded = vocab.encode_row(row)
        pairs.append(
            {
                "control": control.astype(np.float32, copy=False),
                "treated": prototypes[condition_id].astype(np.float32, copy=False),
                "perturbation_id": int(encoded["perturbation_id"]),
                "cell_line_id": int(encoded["cell_line_id"]),
                "perturbation": normalize_value(row.get("perturbation")),
                "cell_line": normalize_value(row.get("cell_line")),
                "dose": float(encoded["dose"]),
                "time": float(encoded["time"]),
            }
        )
    return pairs, vocab


def _optional_path(value: object) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
