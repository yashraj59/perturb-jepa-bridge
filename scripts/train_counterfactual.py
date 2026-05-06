from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.models.counterfactual import (
    CounterfactualResponsePredictor,
    PerturbationConditionEncoder,
    counterfactual_mmd_loss,
    gaussian_nll_loss,
)
from perturb_jepa.training.seed import seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Perturb-JEPA counterfactual heads/objectives.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated synthetic batches.")
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    if not args.synthetic:
        print("No real-data counterfactual loader is configured yet; running synthetic scaffold. Pass --synthetic to make this explicit.")
    _run_synthetic_counterfactual(config, checkpoint_out=args.checkpoint_out)
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


def _load_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig.smoke()
    if path.suffix.lower() == ".json":
        return ExperimentConfig.load_json(path)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML; use JSON or install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        return ExperimentConfig.from_dict(yaml.safe_load(handle) or {})


if __name__ == "__main__":
    raise SystemExit(main())
