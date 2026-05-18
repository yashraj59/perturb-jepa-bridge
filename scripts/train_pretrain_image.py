from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.data.images import ImageManifestCollator, ImageManifestDataset
from perturb_jepa.losses import masked_cosine_jepa_loss
from perturb_jepa.models.image_encoder import patchify
from perturb_jepa.training.checkpoint import save_checkpoint
from perturb_jepa.training.real_data import (
    infer_image_shape_from_array,
    load_yaml_or_json_config,
    override_bridge_config_for_real_data,
    raw_get,
    resize_for_manifest,
)
from perturb_jepa.training.seed import seed_everything
from scripts.train_bridge import _run_synthetic_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain the image encoder scaffold.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated synthetic image batches.")
    parser.add_argument("--image-manifest", type=Path, default=None, help="CSV image manifest for real image pretraining.")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config, raw_config = load_yaml_or_json_config(args.config)
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    manifest_path = args.image_manifest or _optional_path(raw_get(raw_config, ("data", "image_manifest")))
    if args.synthetic or manifest_path is None:
        if not args.synthetic:
            print("No image manifest provided; running synthetic pretrain. Pass --image-manifest for real data.")
        _run_synthetic_training(config, stage="pretrain_image", checkpoint_out=args.checkpoint_out)
    else:
        _run_real_image_pretraining(
            config,
            manifest_path=manifest_path,
            image_root=args.image_root or _optional_path(raw_get(raw_config, ("data", "image_root"))) or Path(""),
            max_images=args.max_images or raw_get(raw_config, ("data", "max_images")),
            batch_size=args.batch_size or raw_get(raw_config, ("training", "batch_size"), 32),
            checkpoint_out=args.checkpoint_out,
        )
    return 0


def _run_real_image_pretraining(
    config: ExperimentConfig,
    *,
    manifest_path: Path,
    image_root: Path,
    max_images: int | None,
    batch_size: int,
    checkpoint_out: Path | None,
) -> None:
    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    manifest = pd.read_csv(manifest_path)
    if max_images is not None and max_images > 0:
        manifest = manifest.iloc[:max_images].copy()
    dataset = ImageManifestDataset(
        manifest,
        image_root=image_root,
        channels=config.model.image.in_channels,
        resize=resize_for_manifest(manifest, config.model.image.image_size),
    )
    first = dataset[0].image
    channels, image_size = infer_image_shape_from_array(first)
    config = override_bridge_config_for_real_data(
        config,
        image_channels=channels,
        image_size=image_size,
    )
    model = config.build_model().to(device)
    optimizer = config.build_optimizer(
        [
            *model.image_encoder.parameters(),
            *model.image_jepa_predictor.parameters(),
        ]
    )
    collator = ImageManifestCollator(
        patch_size=config.model.image.patch_size,
        patch_mask_prob=0.15,
        seed=config.training.seed,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, collate_fn=collator)
    global_step = 0
    model.train()
    for _ in range(max(1, config.training.steps)):
        for batch in loader:
            images = batch.images.to(device)
            patch_mask = batch.image_patch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model.image_encoder(images, patch_mask=patch_mask)
            with torch.no_grad(), model._set_eval_temporarily((model.image_teacher,)):
                teacher = model.image_teacher(images, patch_mask=None)
            patch_prediction = model.image_jepa_predictor(output.patch_embeddings)
            target = patchify(images, config.model.image.patch_size)
            target_mask = patch_mask if patch_mask.any() else torch.ones_like(patch_mask)
            image_mask_loss = torch.nn.functional.mse_loss(output.patch_reconstruction[target_mask], target[target_mask])
            image_jepa_loss = masked_cosine_jepa_loss(patch_prediction, teacher.patch_embeddings, patch_mask)
            loss = config.loss.image_mask * image_mask_loss + config.loss.jepa * image_jepa_loss
            loss.backward()
            if config.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [*model.image_encoder.parameters(), *model.image_jepa_predictor.parameters()],
                    config.training.grad_clip_norm,
                )
            optimizer.step()
            model.update_teachers(decay=config.training.ema_decay)
            if config.training.log_every > 0 and global_step % config.training.log_every == 0:
                print(
                    f"stage=pretrain_image step={global_step} "
                    f"image_mask={float(image_mask_loss.detach().cpu()):.4f} "
                    f"image_jepa={float(image_jepa_loss.detach().cpu()):.4f} "
                    f"total={float(loss.detach().cpu()):.4f}"
                )
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
            trainer_state={"global_step": global_step, "images": len(dataset)},
            experiment_config=config,
            metadata={"stage": "pretrain_image", "image_manifest": str(manifest_path)},
        )


def _optional_path(value: object) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
