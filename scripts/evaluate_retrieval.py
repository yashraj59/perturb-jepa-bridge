from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.baselines.batch_only_baseline import batch_only_retrieval_metrics
from perturb_jepa.baselines.metadata_only_retrieval import metadata_only_retrieval_metrics
from perturb_jepa.config import ExperimentConfig
from perturb_jepa.data.condition_bags import ImageConditionBagDataset, PairedConditionBagDataset, RNAConditionBagDataset
from perturb_jepa.data.conditions import MetadataVocab
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.training.checkpoint import load_checkpoint
from perturb_jepa.training.real_data import (
    metadata_tensors,
    prepare_expression_matrix,
    raw_get,
    read_h5ad_subset,
    resize_for_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate learned cross-modal retrieval embeddings.")
    parser.add_argument("--rna-embeddings", type=Path)
    parser.add_argument("--image-embeddings", type=Path)
    parser.add_argument("--rna-metadata", type=Path)
    parser.add_argument("--image-metadata", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Bridge checkpoint for real-data embedding extraction.")
    parser.add_argument("--rna-anndata", type=Path)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--image-root", type=Path, default=Path(""))
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-embeddings-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Run on a deterministic synthetic dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", type=int, default=8)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        rna_embeddings, image_embeddings, rna_metadata, image_metadata = _synthetic_data(
            seed=args.seed,
            conditions=args.conditions,
            dim=args.dim,
        )
    elif args.checkpoint is not None:
        if args.rna_anndata is None or args.image_manifest is None:
            raise ValueError("--checkpoint evaluation requires --rna-anndata and --image-manifest")
        rna_embeddings, image_embeddings, rna_metadata, image_metadata = _embeddings_from_checkpoint(
            checkpoint_path=args.checkpoint,
            rna_path=args.rna_anndata,
            manifest_path=args.image_manifest,
            image_root=args.image_root,
            max_cells=args.max_cells,
            max_images=args.max_images,
            n_top_genes=args.n_top_genes,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )
        if args.save_embeddings_dir is not None:
            args.save_embeddings_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.save_embeddings_dir / "rna_embeddings.npy", rna_embeddings)
            np.save(args.save_embeddings_dir / "image_embeddings.npy", image_embeddings)
            rna_metadata.to_csv(args.save_embeddings_dir / "rna_metadata.csv", index=False)
            image_metadata.to_csv(args.save_embeddings_dir / "image_metadata.csv", index=False)
    else:
        required = (args.rna_embeddings, args.image_embeddings, args.rna_metadata, args.image_metadata)
        if any(value is None for value in required):
            raise ValueError("provide RNA/image embeddings and metadata, or pass --synthetic")
        rna_embeddings = np.load(args.rna_embeddings)
        image_embeddings = np.load(args.image_embeddings)
        rna_metadata = pd.read_csv(args.rna_metadata)
        image_metadata = pd.read_csv(args.image_metadata)

    rows = [
        {
            "method": "learned_embeddings",
            **cross_modal_retrieval_metrics(rna_embeddings, image_embeddings, rna_metadata, image_metadata),
        },
        {
            "method": "metadata_only",
            **metadata_only_retrieval_metrics(rna_metadata, image_metadata),
        },
        {
            "method": "batch_only",
            **batch_only_retrieval_metrics(rna_metadata, image_metadata),
        },
    ]
    report = pd.DataFrame.from_records(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    else:
        print(report.to_csv(index=False), end="")
    return 0


def _embeddings_from_checkpoint(
    *,
    checkpoint_path: Path,
    rna_path: Path,
    manifest_path: Path,
    image_root: Path,
    max_cells: int | None,
    max_images: int | None,
    n_top_genes: int | None,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if not checkpoint.get("experiment_config"):
        raise ValueError("checkpoint does not contain experiment_config")
    config = ExperimentConfig.from_dict(checkpoint["experiment_config"])
    adata = read_h5ad_subset(rna_path, max_cells=max_cells, seed=config.training.seed)
    expression, gene_token_ids = prepare_expression_matrix(
        adata.X,
        n_top_genes=n_top_genes,
        max_genes=config.model.rna.max_genes,
    )
    rna_metadata = pd.DataFrame(adata.obs).reset_index(drop=False).rename(columns={"index": "sample_id"})
    image_manifest = pd.read_csv(manifest_path)
    if max_images is not None and max_images > 0:
        image_manifest = image_manifest.iloc[:max_images].copy()
    vocab = MetadataVocab.from_frames([rna_metadata, image_manifest])
    model = config.build_model().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    rna_bags = RNAConditionBagDataset(
        expression,
        rna_metadata,
        rna_bag_size=raw_get(checkpoint["experiment_config"], ("data", "rna_bag_size"), 128),
        min_rna_bag_size=1,
        split="test",
        seed=config.training.seed,
    )
    image_bags = ImageConditionBagDataset(
        image_manifest,
        image_bag_size=raw_get(checkpoint["experiment_config"], ("data", "image_bag_size"), 128),
        min_image_bag_size=1,
        split="test",
        seed=config.training.seed,
        image_root=image_root,
        channels=config.model.image.in_channels,
        resize=resize_for_manifest(image_manifest, config.model.image.image_size),
    )
    paired = PairedConditionBagDataset(rna_bags, image_bags)
    collate = _EvalBagCollator(gene_token_ids, vocab, device=device)
    loader = DataLoader(paired, batch_size=int(batch_size), shuffle=False, collate_fn=collate)
    rna_embeddings: list[np.ndarray] = []
    image_embeddings: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                gene_ids=batch["gene_ids"],
                expression_values=batch["expression_values"],
                rna_token_mask=None,
                rna_bag_mask=batch["rna_bag_mask"],
                images=batch["images"],
                image_patch_mask=None,
                image_bag_mask=batch["image_bag_mask"],
                **batch["metadata"],
            )
            rna_embeddings.append(outputs["rna_shared"].detach().cpu().numpy())
            image_embeddings.append(outputs["image_shared"].detach().cpu().numpy())
            rows.extend(batch["rows"])
    metadata = pd.DataFrame(rows)
    if "condition_key" not in metadata.columns:
        metadata["condition_key"] = metadata["condition_id"]
    return (
        np.concatenate(rna_embeddings, axis=0),
        np.concatenate(image_embeddings, axis=0),
        metadata.copy(),
        metadata.copy(),
    )


class _EvalBagCollator:
    def __init__(self, gene_token_ids: np.ndarray, metadata_vocab: MetadataVocab, *, device: torch.device) -> None:
        self.gene_token_ids = np.asarray(gene_token_ids, dtype=np.int64)
        self.metadata_vocab = metadata_vocab
        self.device = device

    def __call__(self, items: list[dict]) -> dict[str, object]:
        rna_arrays = [np.asarray(item["rna"]["x"], dtype=np.float32) for item in items]
        image_arrays = [np.asarray(item["image"]["x"], dtype=np.float32) for item in items]
        max_rna = max(array.shape[0] for array in rna_arrays)
        max_image = max(array.shape[0] for array in image_arrays)
        expression_np = np.zeros((len(items), max_rna, rna_arrays[0].shape[-1]), dtype=np.float32)
        images_np = np.zeros((len(items), max_image, *image_arrays[0].shape[1:]), dtype=np.float32)
        rna_mask_np = np.zeros((len(items), max_rna), dtype=bool)
        image_mask_np = np.zeros((len(items), max_image), dtype=bool)
        for index, array in enumerate(rna_arrays):
            expression_np[index, : array.shape[0]] = array
            rna_mask_np[index, : array.shape[0]] = True
        for index, array in enumerate(image_arrays):
            images_np[index, : array.shape[0]] = array
            image_mask_np[index, : array.shape[0]] = True
        expression = torch.as_tensor(expression_np, dtype=torch.float32, device=self.device)
        images = torch.as_tensor(images_np, dtype=torch.float32, device=self.device)
        gene_ids = torch.as_tensor(
            np.broadcast_to(self.gene_token_ids[None, None, :], expression.shape).copy(),
            dtype=torch.long,
            device=self.device,
        )
        rows = [item["condition"] for item in items]
        return {
            "gene_ids": gene_ids,
            "expression_values": expression,
            "rna_bag_mask": torch.as_tensor(rna_mask_np, dtype=torch.bool, device=self.device),
            "images": images,
            "image_bag_mask": torch.as_tensor(image_mask_np, dtype=torch.bool, device=self.device),
            "metadata": metadata_tensors(rows, self.metadata_vocab, device=self.device),
            "rows": rows,
        }


def _synthetic_data(*, seed: int, conditions: int, dim: int):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(conditions, dim)).astype(np.float32)
    rna = base + 0.03 * rng.normal(size=base.shape).astype(np.float32)
    image = base + 0.03 * rng.normal(size=base.shape).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "condition_key": [f"drug{i}|{i % 3 + 1}uM|{24 + 24 * (i % 2)}h|CL{i % 2}" for i in range(conditions)],
            "perturbation": [f"drug{i}" for i in range(conditions)],
            "dose": [f"{i % 3 + 1}uM" for i in range(conditions)],
            "time": [f"{24 + 24 * (i % 2)}h" for i in range(conditions)],
            "cell_line": [f"CL{i % 2}" for i in range(conditions)],
            "moa": [f"moa{i % 3}" for i in range(conditions)],
            "batch": [f"batch{i % 2}" for i in range(conditions)],
            "plate": [f"plate{i % 3}" for i in range(conditions)],
            "run": [f"run{i % 2}" for i in range(conditions)],
            "sequencing_lane": [f"L{i % 2}" for i in range(conditions)],
            "library_id": [f"lib{i % 4}" for i in range(conditions)],
        }
    )
    return rna, image, metadata.copy(), metadata.copy()


if __name__ == "__main__":
    raise SystemExit(main())
