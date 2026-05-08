from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.image_counterfactual import image_counterfactual_metrics
from perturb_jepa.evaluation.rna_counterfactual import rna_counterfactual_metrics
from perturb_jepa.data.conditions import MetadataVocab
from perturb_jepa.models.counterfactual import CounterfactualResponsePredictor, PerturbationConditionEncoder
from perturb_jepa.training.real_data import prepare_expression_matrix, read_h5ad_subset
from scripts.train_counterfactual import _control_treated_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RNA and image counterfactual predictions.")
    parser.add_argument("--predicted-rna", type=Path)
    parser.add_argument("--observed-rna", type=Path)
    parser.add_argument("--control-rna", type=Path)
    parser.add_argument("--predicted-image-embeddings", type=Path)
    parser.add_argument("--observed-image-embeddings", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Counterfactual checkpoint for real RNA evaluation.")
    parser.add_argument("--rna-anndata", type=Path)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--control-values", default="control,ctrl,dmso,vehicle,untreated,mock")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict-vocab", action="store_true")
    parser.add_argument("--save-arrays-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Run on a deterministic synthetic dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", type=int, default=8)
    parser.add_argument("--genes", type=int, default=32)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    include_image = True
    if args.synthetic:
        predicted_rna, observed_rna, control_rna, predicted_image, observed_image, metadata = _synthetic_data(
            seed=args.seed,
            conditions=args.conditions,
            genes=args.genes,
            dim=args.dim,
        )
    elif args.checkpoint is not None:
        if args.rna_anndata is None:
            raise ValueError("--checkpoint evaluation requires --rna-anndata")
        predicted_rna, observed_rna, control_rna, metadata = _rna_predictions_from_checkpoint(
            checkpoint_path=args.checkpoint,
            rna_path=args.rna_anndata,
            max_cells=args.max_cells,
            n_top_genes=args.n_top_genes,
            control_values=[value.strip() for value in args.control_values.split(",") if value.strip()],
            device=torch.device(args.device),
            strict_vocab=args.strict_vocab,
        )
        predicted_image = observed_image = None
        include_image = False
        if args.save_arrays_dir is not None:
            args.save_arrays_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.save_arrays_dir / "predicted_rna.npy", predicted_rna)
            np.save(args.save_arrays_dir / "observed_rna.npy", observed_rna)
            np.save(args.save_arrays_dir / "control_rna.npy", control_rna)
            metadata.to_csv(args.save_arrays_dir / "counterfactual_metadata.csv", index=False)
    else:
        required = (
            args.predicted_rna,
            args.observed_rna,
            args.control_rna,
            args.predicted_image_embeddings,
            args.observed_image_embeddings,
            args.metadata,
        )
        if any(value is None for value in required):
            raise ValueError("provide prediction arrays and metadata, or pass --synthetic")
        predicted_rna = np.load(args.predicted_rna)
        observed_rna = np.load(args.observed_rna)
        control_rna = np.load(args.control_rna)
        predicted_image = np.load(args.predicted_image_embeddings)
        observed_image = np.load(args.observed_image_embeddings)
        metadata = pd.read_csv(args.metadata)

    metrics = rna_counterfactual_metrics(predicted_rna, observed_rna, control_rna, metadata)
    if include_image:
        metrics.update(image_counterfactual_metrics(predicted_image, observed_image, metadata))
    report = pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    else:
        print(report.to_csv(index=False), end="")
    return 0


def _rna_predictions_from_checkpoint(
    *,
    checkpoint_path: Path,
    rna_path: Path,
    max_cells: int | None,
    n_top_genes: int | None,
    control_values: list[str],
    device: torch.device,
    strict_vocab: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get("experiment_config") or {}
    max_genes = (((config_dict.get("model") or {}).get("rna") or {}).get("max_genes"))
    adata = read_h5ad_subset(rna_path, max_cells=max_cells, seed=((config_dict.get("training") or {}).get("seed", 0)))
    expression, _ = prepare_expression_matrix(adata.X, n_top_genes=n_top_genes, max_genes=max_genes)
    metadata = pd.DataFrame(adata.obs).reset_index(drop=True)
    checkpoint_metadata = checkpoint.get("metadata") or {}
    if "metadata_vocab" not in checkpoint_metadata:
        raise ValueError("checkpoint metadata is missing saved metadata_vocab; refusing to rebuild vocab from eval data")
    vocab = MetadataVocab.from_dict(checkpoint_metadata["metadata_vocab"])
    pairs, vocab = _control_treated_pairs(
        expression,
        metadata,
        control_values,
        metadata_vocab=vocab,
        strict_vocab=strict_vocab,
    )
    if not pairs:
        raise ValueError("No control-relative condition pairs found for evaluation")
    predictor_first = checkpoint["predictor"]["net.0.weight"]
    prototype_dim = int(checkpoint["predictor"]["net.4.weight"].shape[0] // 2)
    condition_dim = int(predictor_first.shape[1] - prototype_dim)
    condition_encoder = PerturbationConditionEncoder(
        num_perturbations=vocab.num_perturbations,
        num_cell_lines=vocab.num_cell_lines,
        hidden_dim=condition_dim,
        output_dim=condition_dim,
    ).to(device)
    predictor = CounterfactualResponsePredictor(
        prototype_dim=prototype_dim,
        condition_dim=condition_dim,
        hidden_dim=int(predictor_first.shape[0]),
    ).to(device)
    condition_encoder.load_state_dict(checkpoint["condition_encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    condition_encoder.eval()
    predictor.eval()
    predicted: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    control_values_out: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for pair in pairs:
            control = torch.as_tensor(pair["control"], dtype=torch.float32, device=device).reshape(1, 1, -1)
            condition_embedding = condition_encoder(
                perturbation_id=torch.tensor([pair["perturbation_id"]], dtype=torch.long, device=device),
                cell_line_id=torch.tensor([pair["cell_line_id"]], dtype=torch.long, device=device),
                dose=torch.tensor([pair["dose"]], dtype=torch.float32, device=device),
                time=torch.tensor([pair["time"]], dtype=torch.float32, device=device),
            )
            output = predictor(control, condition_embedding)
            predicted.append(output.treated_mu.squeeze(0).squeeze(0).detach().cpu().numpy())
            observed.append(np.asarray(pair["treated"], dtype=np.float32))
            control_values_out.append(np.asarray(pair["control"], dtype=np.float32))
            rows.append(
                {
                    "perturbation": str(pair.get("perturbation", "")),
                    "dose": pair["dose"],
                    "time": pair["time"],
                    "cell_line": str(pair.get("cell_line", "")),
                }
            )
    return np.stack(predicted), np.stack(observed), np.stack(control_values_out), pd.DataFrame(rows)


def _synthetic_data(*, seed: int, conditions: int, genes: int, dim: int):
    rng = np.random.default_rng(seed)
    control = rng.normal(size=(conditions, genes)).astype(np.float32)
    effect = rng.normal(scale=0.4, size=(conditions, genes)).astype(np.float32)
    observed = control + effect + rng.normal(scale=0.03, size=(conditions, genes)).astype(np.float32)
    predicted = control + effect + rng.normal(scale=0.05, size=(conditions, genes)).astype(np.float32)
    image_base = rng.normal(size=(conditions, dim)).astype(np.float32)
    image_observed = image_base + rng.normal(scale=0.03, size=(conditions, dim)).astype(np.float32)
    image_predicted = image_base + rng.normal(scale=0.05, size=(conditions, dim)).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "condition_key": [f"drug{i}|{i % 3 + 1}uM|{24 + 24 * (i % 2)}h|CL{i % 2}" for i in range(conditions)],
            "perturbation": [f"drug{i}" for i in range(conditions)],
            "dose": [f"{i % 3 + 1}uM" for i in range(conditions)],
            "time": [f"{24 + 24 * (i % 2)}h" for i in range(conditions)],
            "cell_line": [f"CL{i % 2}" for i in range(conditions)],
            "moa": [f"moa{i % 3}" for i in range(conditions)],
        }
    )
    return predicted, observed, control, image_predicted, image_observed, metadata


if __name__ == "__main__":
    raise SystemExit(main())
