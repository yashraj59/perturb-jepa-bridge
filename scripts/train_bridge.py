from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.data.condition_bags import (
    ImageConditionBagDataset,
    PairedConditionBagDataset,
    RNAConditionBagDataset,
    summarize_technical_metadata,
)
from perturb_jepa.data.conditions import MetadataVocab
from perturb_jepa.models.image_encoder import patchify
from perturb_jepa.losses import BridgeLossWeights, bridge_loss
from perturb_jepa.training.checkpoint import load_checkpoint, save_checkpoint
from perturb_jepa.training.objectives import build_uncertainty_weighting, weighted_bridge_total
from perturb_jepa.training.real_data import (
    assign_real_data_splits,
    filter_metadata_by_split,
    load_yaml_or_json_config,
    make_token_mask,
    metadata_tensors,
    override_bridge_config_for_real_data,
    parse_heldout_values,
    prepare_expression_matrix,
    raw_get,
    read_h5ad_subset,
    resize_for_manifest,
)
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import BridgeTrainer, forward_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the cross-modal Perturb-JEPA bridge.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated synthetic batches.")
    parser.add_argument("--rna-anndata", type=Path, default=None)
    parser.add_argument("--image-manifest", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split-col", default=None)
    parser.add_argument("--train-split-value", default=None)
    parser.add_argument("--eval-split-value", default=None)
    parser.add_argument(
        "--split-strategy",
        choices=[
            "none",
            "random_grouped",
            "heldout_batch",
            "heldout_perturbation",
            "heldout_dose_time",
            "heldout_cell_line",
            "heldout_moa",
        ],
        default=None,
    )
    parser.add_argument("--heldout-values", default=None)
    parser.add_argument("--heldout-fraction", type=float, default=None)
    parser.add_argument("--rna-mask-prob", type=float, default=None)
    parser.add_argument("--image-patch-mask-prob", type=float, default=None)
    parser.add_argument(
        "--rna-pretrain-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose rna_encoder weights initialize bridge training.",
    )
    parser.add_argument(
        "--image-pretrain-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose image_encoder weights initialize bridge training.",
    )
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config, raw_config = load_yaml_or_json_config(args.config)
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    data_updates = {}
    for attr, value in (
        ("split_col", args.split_col),
        ("train_split_value", args.train_split_value),
        ("eval_split_value", args.eval_split_value),
        ("split_strategy", args.split_strategy),
        ("heldout_fraction", args.heldout_fraction),
        ("rna_mask_prob", args.rna_mask_prob),
        ("image_patch_mask_prob", args.image_patch_mask_prob),
    ):
        if value is not None:
            data_updates[attr] = value
    if args.heldout_values is not None:
        data_updates["heldout_values"] = tuple(
            parse_heldout_values(args.heldout_values, strategy=args.split_strategy or config.data.split_strategy) or ()
        )
    if data_updates:
        config = replace(config, data=replace(config.data, **data_updates))
    rna_path = args.rna_anndata or _optional_path(config.data.rna_anndata or raw_get(raw_config, ("data", "rna_anndata")))
    manifest_path = args.image_manifest or _optional_path(
        config.data.image_manifest or raw_get(raw_config, ("data", "image_manifest"))
    )
    if args.synthetic or rna_path is None or manifest_path is None:
        if not args.synthetic:
            print("Real bridge training needs both --rna-anndata and --image-manifest; running synthetic scaffold.")
        _run_synthetic_training(config, stage="bridge", checkpoint_out=args.checkpoint_out)
    else:
        _run_real_bridge_training(
            config,
            raw_config=raw_config,
            rna_path=rna_path,
            manifest_path=manifest_path,
            image_root=args.image_root or _optional_path(config.data.image_root or raw_get(raw_config, ("data", "image_root"))) or Path(""),
            max_cells=args.max_cells or raw_get(raw_config, ("data", "max_cells")),
            max_images=args.max_images or raw_get(raw_config, ("data", "max_images")),
            n_top_genes=args.n_top_genes or raw_get(raw_config, ("data", "n_top_genes")),
            batch_size=args.batch_size or config.training.batch_size,
            rna_pretrain_checkpoint=args.rna_pretrain_checkpoint,
            image_pretrain_checkpoint=args.image_pretrain_checkpoint,
            checkpoint_out=args.checkpoint_out,
        )
    return 0


def _run_synthetic_training(
    config: ExperimentConfig,
    *,
    stage: str,
    checkpoint_out: Path | None,
) -> None:
    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=config.loss,
        schedule=config.training.objective_schedule,
        uncertainty_weighting=config.training.uncertainty_weighting,
        ema_decay=config.training.ema_decay,
        device=device,
        grad_clip_norm=config.training.grad_clip_norm,
    )
    for step in range(config.training.steps):
        batch = make_synthetic_bridge_batch(**asdict(config.synthetic), device=device)
        terms = trainer.step(batch)
        if config.training.log_every > 0 and (step + 1) % config.training.log_every == 0:
            print(f"stage={stage} step={step} " + _format_terms(terms))
    if checkpoint_out is not None:
        save_checkpoint(
            checkpoint_out,
            model=model,
            optimizer=optimizer,
            trainer_state=trainer.state_dict(),
            experiment_config=config,
            metadata={"stage": stage},
        )


def _synthetic_stage_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    *,
    grad_clip_norm: float | None,
    ema_decay: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = forward_batch(model, batch)
    terms: dict[str, torch.Tensor] = {}
    if "rna_reconstruction" in outputs:
        terms["rna_mask"] = torch.nn.functional.mse_loss(outputs["rna_reconstruction"], batch.expression_values)
    if "image_patch_reconstruction" in outputs:
        prediction = outputs["image_patch_reconstruction"]
        target = patchify(batch.images, model.config.image.patch_size)
        if prediction.ndim == target.ndim + 1 and prediction.shape[1] == 1:
            prediction = prediction.squeeze(1)
        terms["image_mask"] = torch.nn.functional.mse_loss(prediction, target)
    if "rna_shared" in outputs and "image_shared" in outputs:
        terms["align"] = 1.0 - torch.nn.functional.cosine_similarity(
            outputs["rna_shared"],
            outputs["image_shared"],
            dim=-1,
        ).mean()
    if not terms:
        raise RuntimeError("model forward pass produced no synthetic training terms")
    total = sum(terms.values())
    total.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    if hasattr(model, "update_teachers"):
        model.update_teachers(decay=ema_decay)
    terms["total"] = total.detach()
    return {name: float(value.detach().cpu()) for name, value in terms.items()}


def _run_real_bridge_training(
    config: ExperimentConfig,
    *,
    raw_config: dict,
    rna_path: Path,
    manifest_path: Path,
    image_root: Path,
    max_cells: int | None,
    max_images: int | None,
    n_top_genes: int | None,
    batch_size: int,
    rna_pretrain_checkpoint: Path | None,
    image_pretrain_checkpoint: Path | None,
    checkpoint_out: Path | None,
) -> None:
    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    adata = read_h5ad_subset(rna_path, max_cells=max_cells, seed=config.training.seed)
    expression, gene_token_ids = prepare_expression_matrix(
        adata.X,
        n_top_genes=n_top_genes,
        max_genes=config.model.rna.max_genes,
        normalize=config.data.rna_normalize,
    )
    rna_metadata = pd.DataFrame(adata.obs).reset_index(drop=False).rename(columns={"index": "sample_id"})
    rna_metadata["_matrix_index"] = np.arange(len(rna_metadata))
    image_manifest = pd.read_csv(manifest_path)
    if max_images is not None and max_images > 0:
        image_manifest = image_manifest.iloc[:max_images].copy()
    rna_metadata, image_manifest, split_metadata = _split_real_metadata(config, rna_metadata, image_manifest)
    rna_train_metadata = filter_metadata_by_split(
        rna_metadata,
        split_col=config.data.split_col,
        split_value=config.data.train_split_value,
        name="RNA train",
    )
    image_train_manifest = filter_metadata_by_split(
        image_manifest,
        split_col=config.data.split_col,
        split_value=config.data.train_split_value,
        name="image train",
    )
    expression = expression[rna_train_metadata["_matrix_index"].astype(int).to_numpy()]
    rna_train_metadata = rna_train_metadata.drop(columns=["_matrix_index"])
    metadata_vocab = MetadataVocab.from_frames([rna_train_metadata, image_train_manifest])
    config = override_bridge_config_for_real_data(
        config,
        num_genes=expression.shape[1],
        max_genes=expression.shape[1],
        metadata_vocab=metadata_vocab,
    )
    rna_bags = RNAConditionBagDataset(
        expression,
        rna_train_metadata,
        rna_bag_size=config.data.rna_bag_size,
        min_rna_bag_size=config.data.min_rna_bag_size,
        split="train",
        seed=config.training.seed,
        condition_key_col=config.data.condition_key,
        balanced_sample_col=config.data.bag_sample_tech_col,
    )
    image_bags = ImageConditionBagDataset(
        image_train_manifest,
        image_bag_size=config.data.image_bag_size,
        min_image_bag_size=config.data.min_image_bag_size,
        split="train",
        seed=config.training.seed,
        image_root=image_root,
        channels=config.model.image.in_channels,
        resize=resize_for_manifest(image_manifest, config.model.image.image_size),
        condition_key_col=config.data.condition_key,
        balanced_sample_col=config.data.bag_sample_tech_col,
    )
    paired = PairedConditionBagDataset(rna_bags, image_bags)
    if len(paired) == 0:
        raise ValueError("No overlapping biological conditions between RNA and image data")
    print(f"Real bridge conditions: {len(paired)} shared biological condition bags")
    model = config.build_model().to(device)
    pretrain_metadata: dict[str, object] = {}
    if rna_pretrain_checkpoint is not None:
        pretrain_metadata["rna"] = _load_pretrained_encoder(
            model,
            rna_pretrain_checkpoint,
            modality="rna",
            device=device,
        )
    if image_pretrain_checkpoint is not None:
        pretrain_metadata["image"] = _load_pretrained_encoder(
            model,
            image_pretrain_checkpoint,
            modality="image",
            device=device,
        )
    optimizer = config.build_optimizer(model.parameters())
    uncertainty_weighting = build_uncertainty_weighting(config.training.uncertainty_weighting)
    if uncertainty_weighting is not None:
        uncertainty_weighting.to(device)
        optimizer.add_param_group({"params": list(uncertainty_weighting.parameters())})
    collate = _BridgeBagCollator(
        gene_token_ids,
        metadata_vocab,
        device=device,
        rna_mask_prob=config.data.rna_mask_prob,
        image_patch_mask_prob=config.data.image_patch_mask_prob,
        image_patch_size=config.model.image.patch_size,
        technical_summary=config.data.technical_summary,
    )
    loader = DataLoader(paired, batch_size=int(batch_size), shuffle=True, collate_fn=collate)
    global_step = 0
    for _ in range(max(1, config.training.steps)):
        for batch in loader:
            model.train()
            if uncertainty_weighting is not None:
                uncertainty_weighting.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                gene_ids=batch["gene_ids"],
                expression_values=batch["expression_values"],
                rna_token_mask=batch["rna_token_mask"],
                rna_bag_mask=batch["rna_bag_mask"],
                images=batch["images"],
                image_patch_mask=batch["image_patch_mask"],
                image_bag_mask=batch["image_bag_mask"],
                **batch["metadata"],
            )
            image_target = patchify(
                batch["images"].reshape(-1, *batch["images"].shape[-3:]),
                config.model.image.patch_size,
            ).reshape(*batch["images"].shape[:2], -1, config.model.image.patch_dim)
            _, terms = bridge_loss(
                outputs,
                rna_values=batch["expression_values"],
                rna_mask=batch["rna_token_mask"],
                image_patches=image_target,
                image_patch_mask=batch["image_patch_mask"],
                bio_keys=batch["bio_keys"],
                perturbation_id=batch["metadata"]["perturbation_id"],
                batch_id=batch["metadata"]["batch_id"],
                rna_batch_id=batch["rna_batch_id"],
                image_batch_id=batch["image_batch_id"],
                temperature=config.loss.temperature,
                weights=BridgeLossWeights(),
            )
            raw_terms = {name: value for name, value in terms.items() if name != "total"}
            total, objective_terms = weighted_bridge_total(
                raw_terms,
                weights=config.loss,
                schedule=config.training.objective_schedule,
                step=global_step,
                uncertainty_weighting=uncertainty_weighting,
            )
            terms = {**raw_terms, **objective_terms}
            total.backward()
            if config.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            model.update_teachers(decay=config.training.ema_decay)
            if config.training.log_every > 0 and global_step % config.training.log_every == 0:
                floats = {name: float(value.detach().cpu()) for name, value in terms.items()}
                print(f"stage=bridge step={global_step} " + _format_terms(floats))
            global_step += 1
            if global_step >= config.training.steps:
                break
        if global_step >= config.training.steps:
            break
    if checkpoint_out is not None:
        trainer_state = {"global_step": global_step, "conditions": len(paired)}
        if uncertainty_weighting is not None:
            trainer_state["uncertainty_weighting"] = uncertainty_weighting.state_dict()
        save_checkpoint(
            checkpoint_out,
            model=model,
            optimizer=optimizer,
            trainer_state=trainer_state,
            experiment_config=config,
            metadata={
                "stage": "bridge",
                "rna_anndata": str(rna_path),
                "image_manifest": str(manifest_path),
                "pretrain_checkpoints": pretrain_metadata,
                "metadata_vocab": metadata_vocab.to_dict(),
                "split": split_metadata,
            },
        )


def _load_pretrained_encoder(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    modality: str,
    device: torch.device,
) -> dict[str, object]:
    if modality not in {"rna", "image"}:
        raise ValueError("modality must be 'rna' or 'image'")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state", {})
    source_prefix = f"{modality}_encoder."
    submodule_state = {
        key.removeprefix(source_prefix): value
        for key, value in model_state.items()
        if key.startswith(source_prefix)
    }
    if not submodule_state:
        raise KeyError(f"{checkpoint_path} has no weights under {source_prefix!r}")

    encoder = getattr(model, f"{modality}_encoder")
    teacher = getattr(model, f"{modality}_teacher")
    encoder.load_state_dict(submodule_state, strict=True)
    teacher.load_state_dict(encoder.state_dict(), strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    print(
        f"Loaded {len(submodule_state)} {modality}_encoder tensors from {checkpoint_path} "
        "and synced the EMA teacher"
    )
    return {
        "path": str(checkpoint_path),
        "loaded_tensors": len(submodule_state),
        "checkpoint_stage": checkpoint.get("metadata", {}).get("stage"),
    }


class _BridgeBagCollator:
    def __init__(
        self,
        gene_token_ids: np.ndarray,
        metadata_vocab: MetadataVocab,
        *,
        device: torch.device,
        rna_mask_prob: float = 0.0,
        image_patch_mask_prob: float = 0.0,
        image_patch_size: int = 8,
        technical_summary: str = "mode",
        strict_vocab: bool = False,
    ) -> None:
        self.gene_token_ids = np.asarray(gene_token_ids, dtype=np.int64)
        self.metadata_vocab = metadata_vocab
        self.device = device
        self.rna_mask_prob = float(rna_mask_prob)
        self.image_patch_mask_prob = float(image_patch_mask_prob)
        self.image_patch_size = int(image_patch_size)
        self.technical_summary = technical_summary
        self.strict_vocab = strict_vocab

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
        rna_tech_rows = [
            summarize_technical_metadata(item["rna"]["tech"], strategy=self.technical_summary) for item in items
        ]
        image_tech_rows = [
            summarize_technical_metadata(item["image"]["tech"], strategy=self.technical_summary) for item in items
        ]
        rna_bag_mask = torch.as_tensor(rna_mask_np, dtype=torch.bool, device=self.device)
        image_bag_mask = torch.as_tensor(image_mask_np, dtype=torch.bool, device=self.device)
        rna_token_mask = make_token_mask(tuple(expression.shape), self.rna_mask_prob, device=self.device)
        rna_token_mask = rna_token_mask & rna_bag_mask.unsqueeze(-1)
        patches = (images.shape[-2] // self.image_patch_size) * (images.shape[-1] // self.image_patch_size)
        image_patch_mask = make_token_mask(
            (len(items), max_image, int(patches)),
            self.image_patch_mask_prob,
            device=self.device,
        )
        image_patch_mask = image_patch_mask & image_bag_mask.unsqueeze(-1)
        return {
            "gene_ids": gene_ids,
            "expression_values": expression,
            "rna_token_mask": rna_token_mask,
            "rna_bag_mask": rna_bag_mask,
            "images": images,
            "image_patch_mask": image_patch_mask,
            "image_bag_mask": image_bag_mask,
            "metadata": metadata_tensors(rows, self.metadata_vocab, device=self.device, strict=self.strict_vocab),
            "rna_batch_id": metadata_tensors(
                rna_tech_rows,
                self.metadata_vocab,
                device=self.device,
                strict=self.strict_vocab,
            )["batch_id"],
            "image_batch_id": metadata_tensors(
                image_tech_rows,
                self.metadata_vocab,
                device=self.device,
                strict=self.strict_vocab,
            )["batch_id"],
            "bio_keys": [item["bio_key"] for item in items],
            "rna_metadata": rna_tech_rows,
            "image_metadata": image_tech_rows,
        }


def _split_real_metadata(
    config: ExperimentConfig,
    rna_metadata: pd.DataFrame,
    image_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if config.data.split_strategy != "none":
        return assign_real_data_splits(
            rna_metadata,
            image_manifest,
            split_strategy=config.data.split_strategy,
            split_col=config.data.split_col,
            train_split_value=config.data.train_split_value,
            eval_split_value=config.data.eval_split_value or "test",
            heldout_values=config.data.heldout_values,
            heldout_fraction=config.data.heldout_fraction,
            seed=config.training.seed,
        )
    rna = rna_metadata.copy()
    image = image_manifest.copy()
    if config.data.split_col in rna.columns and config.data.split_col in image.columns:
        split_metadata = {
            "split_strategy": "none",
            "split_col": config.data.split_col,
            "train_split_value": config.data.train_split_value,
            "eval_split_value": config.data.eval_split_value,
            "heldout_groups": [],
        }
        return rna, image, split_metadata
    rna[config.data.split_col] = config.data.train_split_value
    image[config.data.split_col] = config.data.train_split_value
    return (
        rna,
        image,
        {
            "split_strategy": "none",
            "split_col": config.data.split_col,
            "train_split_value": config.data.train_split_value,
            "eval_split_value": config.data.eval_split_value,
            "heldout_groups": [],
        },
    )


def _load_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig.smoke()
    if path.suffix.lower() == ".json":
        return ExperimentConfig.load_json(path)
    data = _load_yaml(path)
    return ExperimentConfig.from_dict(data)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML; use JSON or install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _format_terms(terms: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in sorted(terms.items()))


def _optional_path(value: object) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
