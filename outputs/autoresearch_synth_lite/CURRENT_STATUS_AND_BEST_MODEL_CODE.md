# Current Status And Best Model Code Bundle

Date: 2026-05-25

This file consolidates the current autoresearch status, the best synthetic-only results, and the implementation entry points for the current best mechanisms. Code blocks are excerpts for review; the source of truth remains the named script or module above each block.

## Executive Status

- Branch: `autoresearch/synthetic-lite-v1`
- Data status: synthetic only; no real data used.
- Compute status: low-compute CPU/tiny synthetic runs only in the documented families below.
- Current protected synthetic geometry model of record: rank-3 train-split-only PLS linear readout installed into bridge raw-linear readout heads.
- Current best counterfactual reference: train-only biological-key condition mean table from Family N.
- Current best count-aware reference: train-only count mean table from Family O.
- No learned JEPA/counterfactual decoder has been promoted over the protected baselines.
- Latest targeted tests: `10 passed in 5.59s`.

## Model Of Record

Source artifact: `outputs/autoresearch_synth_lite/model_of_record.md`

The protected synthetic shared-geometry mechanism is a closed-form rank-3 PLS readout:

- RNA readout: `raw_linear_pseudobulk`
- Image readout: `raw_linear_pooled`
- Tier 2 passed on `synth_micro` seeds `0/1/2`
- Focused Tier 3 passed on `synth_easy_lite` seed `0`
- Focused Tier 3 passed on `synth_batch_confound_lite` seed `0`

Important caveat: this is a protected readout initializer/baseline, not an SGD-trained JEPA model promotion.

## Best Results

### Shared Geometry: Rank-3 PLS Readout

Source artifact: `outputs/autoresearch_synth_lite/diagnostics/PREFIT_PLS_READOUT/TIER2_TIER3_SUMMARY.md`

Tier 2 means on `synth_micro` seeds `0/1/2`:

| metric | value |
|---|---:|
| RNA->image recall@1 | `0.2396 +/- 0.0295` |
| RNA->image recall@5 | `0.6667 +/- 0.1284` |
| RNA latent R2 | `0.5929 +/- 0.1395` |
| Image latent R2 | `0.9134 +/- 0.0206` |
| Batch balanced accuracy | `0.4792 +/- 0.0780` |

### Best Expression-Space Counterfactual Reference: Family N

Source artifact: `outputs/autoresearch_synth_lite/diagnostics/FAMILY_N_DISTILLATION/REPORT.md`

Candidate: `seed2_train_only_condition_mean_table`

| metric | value |
|---|---:|
| Exact train biological-key coverage | `1.0000` |
| Program recovery | `0.7520` |
| Direction accuracy | `0.6899` |
| logFC correlation | `0.7502` |
| Pseudobulk correlation | `0.8725` |
| Top50 DE overlap | `0.5683` |
| Mean delta/target ratio | `0.7400` |

This table uses only `perturbation_id`, `cell_line_id`, `dose`, and `time`; `batch_id` is excluded, and teacher construction uses `0` test target rows.

### Best Count-Aware Reference: Family O

Source artifact: `outputs/autoresearch_synth_lite/diagnostics/FAMILY_O_COUNT_LIKELIHOOD/REPORT.md`

Candidate: `seed2_nb_train_only_count_mean_table`

| metric | value |
|---|---:|
| Exact train biological-key coverage | `1.0000` |
| Program recovery | `0.7433` |
| Direction accuracy | `0.7679` |
| logFC correlation | `0.7562` |
| Pseudobulk correlation | `0.8815` |
| Top50 DE overlap | `0.6392` |
| Poisson NLL | `48.4387` |
| NB NLL | `4.9933` |

Count-aware modeling improved logFC, top50 DE overlap, direction accuracy, and pseudobulk correlation versus Family N, but program recovery dropped from `0.7520` to `0.7433`, so it was not promoted.

## Current Blocker

The exact-condition synthetic split is matching-baseline dominated. Train and test share all biological condition keys on seed 2, so a train-only no-batch condition mean table is extremely strong. Learned neural models can approximate this table, but none has beaten it jointly while preserving program recovery and protected geometry.

The next meaningful benchmark should remove exact target support, for example with held-out biological keys, held-out perturbations, held-out doses, or incomplete train target support. A broader count-MLP sweep on the current exact-key split is not justified.

## Reproduction Commands

```bash
python scripts/run_family_m_transport_baselines.py --dataset synth_micro --seed 2 --rank 3 --device cpu
python scripts/run_family_n_distillation.py --dataset synth_micro --seed 2 --rank 3 --device cpu
python scripts/run_family_o_count_likelihood.py --dataset synth_micro --seed 2 --rank 3 --device cpu
pytest tests/test_family_o_count_likelihood.py tests/test_family_n_distillation.py tests/test_family_m_transport_baselines.py
```

Latest targeted verification:

```text
10 passed in 5.59s
```

## Code Bundle

For a complete source dump, see:

- `outputs/autoresearch_synth_lite/FULL_ARCHITECTURE_CODE_BUNDLE.md`

## Actual End-To-End Architecture Code

This section adds the real bridge architecture and wiring path from configuration to model build, protected PLS installation, trainable clone installation, and counterfactual decoder path. These are the source files that define the architecture currently used by the best protected synthetic setup.

### Script: `perturb_jepa/config.py`

This is the start of the architecture build path. Experiment configs build `PerturbJEPABridge` directly.

```python
def default_bridge_config() -> PerturbJEPABridgeConfig:
    return PerturbJEPABridgeConfig(
        rna=RNAEncoderConfig(vocab_size=128, dim=32, depth=1, heads=4, max_genes=64),
        image=ImageEncoderConfig(in_channels=3, image_size=32, patch_size=8, dim=32, depth=1, heads=4),
        perturbation=PerturbationEncoderConfig(
            num_perturbations=16,
            num_types=3,
            num_cell_lines=4,
            num_batches=4,
            dim=32,
        ),
        shared_dim=32,
        num_bag_prototypes=4,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "synthetic-smoke"
    model: PerturbJEPABridgeConfig = field(default_factory=default_bridge_config)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: BridgeLossWeights = field(default_factory=BridgeLossWeights)
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticBatchConfig = field(default_factory=SyntheticBatchConfig)

    def build_model(self) -> PerturbJEPABridge:
        return PerturbJEPABridge(self.model)
```

### Script: `perturb_jepa/models/bridge.py`

This is the actual JEPA/scRNA/image/perturbation bridge configuration.

```python
@dataclass(frozen=True)
class PerturbJEPABridgeConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int = 128
    num_bag_prototypes: int = 8
    dropout: float = 0.1
    adversary_scale: float = 1.0
    bag_aggregator: str = "attention"
    rna_condition_readout: str = "encoder"
    rna_pseudobulk_normalize: bool = True
    image_condition_readout: str = "encoder"
    image_raw_normalize: bool = True
    counterfactual_rna_residual: bool = False
    counterfactual_rna_program_factorized: bool = False
    counterfactual_rna_num_programs: int = 0
    counterfactual_rna_program_assignment: tuple[int, ...] = ()
    counterfactual_rna_within_program_residual: bool = False
    counterfactual_rna_program_conditioned: bool = False
    counterfactual_rna_program_metadata_context: bool = False
    counterfactual_rna_program_decoder_depth: int = 2
```

### Script: `perturb_jepa/models/bridge.py`

This is the architecture constructor: encoders, EMA teachers, projection heads, PLS-compatible raw-linear heads, bag aggregators, JEPA predictors, perturbation heads, adversaries, and counterfactual modules.

```python
class PerturbJEPABridge(nn.Module):
    def __init__(self, config: PerturbJEPABridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.rna_encoder = RNAEncoder(config.rna)
        self.image_encoder = ImageEncoder(config.image)
        self.rna_teacher = make_ema_teacher(self.rna_encoder)
        self.image_teacher = make_ema_teacher(self.image_encoder)
        self.perturbation_encoder = PerturbationEncoder(config.perturbation)

        self.rna_projection = RNAProjectionHead(config.rna.dim, config.shared_dim, config.shared_dim)
        self.image_projection = ImageProjectionHead(config.image.dim, config.shared_dim, config.shared_dim)
        image_raw_dim = config.image.in_channels * config.image.image_size * config.image.image_size
        self.image_raw_projection = nn.Sequential(
            nn.LayerNorm(image_raw_dim),
            nn.Linear(image_raw_dim, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.image_raw_linear_projection = nn.Linear(image_raw_dim, config.shared_dim)
        self.image_distilled_linear_projection = nn.Linear(image_raw_dim, config.shared_dim)
        self.image_distilled_residual_scale = nn.Parameter(torch.zeros(()))
        self.rna_pseudobulk_projection = nn.Sequential(
            nn.LayerNorm(config.rna.dim),
            nn.Linear(config.rna.dim, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.rna_raw_pseudobulk_projection = nn.Sequential(
            nn.LayerNorm(config.rna.max_genes),
            nn.Linear(config.rna.max_genes, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.rna_raw_linear_projection = nn.Linear(config.rna.max_genes, config.shared_dim)
        self.rna_distilled_linear_projection = nn.Linear(config.rna.max_genes, config.shared_dim)
        self.rna_distilled_residual_scale = nn.Parameter(torch.zeros(()))
        self.rna_teacher_projection = RNAProjectionHead(config.rna.dim, config.shared_dim, config.shared_dim)
        self.image_teacher_projection = ImageProjectionHead(config.image.dim, config.shared_dim, config.shared_dim)
        self.rna_teacher_projection.load_state_dict(self.rna_projection.state_dict())
        self.image_teacher_projection.load_state_dict(self.image_projection.state_dict())

        aggregator_cls = self._bag_aggregator_cls(config.bag_aggregator)
        self.rna_bag_aggregator = aggregator_cls(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_bag_prototypes,
            dropout=config.dropout,
        )
        self.image_bag_aggregator = aggregator_cls(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_bag_prototypes,
            dropout=config.dropout,
        )
        self.rna_teacher_bag_aggregator = make_ema_teacher(self.rna_bag_aggregator)
        self.image_teacher_bag_aggregator = make_ema_teacher(self.image_bag_aggregator)
        self.rna_jepa_predictor = MLP(config.rna.dim, config.rna.dim, config.rna.dim, depth=2, dropout=config.dropout)
        self.image_jepa_predictor = MLP(config.image.dim, config.image.dim, config.image.dim, depth=2, dropout=config.dropout)
        self.state_head = MLP(config.shared_dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.response_head = MLP(config.shared_dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.perturbation_classifier = MLP(
            config.shared_dim,
            config.shared_dim,
            config.perturbation.num_perturbations,
            depth=2,
            dropout=config.dropout,
        )
        self.state_perturbation_adversary = MLP(
            config.shared_dim,
            config.shared_dim,
            config.perturbation.num_perturbations,
            depth=2,
            dropout=config.dropout,
        )
        self.batch_adversary = BatchAdversary(
            config.shared_dim,
            config.perturbation.num_batches,
            hidden_dim=config.shared_dim,
            dropout=config.dropout,
            scale=config.adversary_scale,
        )
        self.delta_gate = MLP(
            config.shared_dim + config.perturbation.dim,
            config.shared_dim,
            config.shared_dim,
            depth=3,
            dropout=config.dropout,
        )
        self.delta_effect = nn.Linear(config.perturbation.dim, config.shared_dim)
        self.rna_distribution_decoder = MLP(
            config.shared_dim,
            config.shared_dim,
            config.rna.vocab_size,
            depth=2,
            dropout=config.dropout,
        )
        if config.counterfactual_rna_program_factorized:
            program_decoder_input_dim = config.shared_dim
            if config.counterfactual_rna_program_conditioned:
                program_decoder_input_dim += config.counterfactual_rna_num_programs
            if config.counterfactual_rna_program_metadata_context:
                program_decoder_input_dim += config.perturbation.num_perturbations + config.perturbation.num_cell_lines + 2
            self.counterfactual_program_decoder = MLP(
                program_decoder_input_dim,
                config.shared_dim,
                config.counterfactual_rna_num_programs,
                depth=config.counterfactual_rna_program_decoder_depth,
                dropout=config.dropout,
            )
            assignment = torch.as_tensor(config.counterfactual_rna_program_assignment, dtype=torch.long)
            self.register_buffer("counterfactual_rna_program_assignment", assignment, persistent=True)
        self.image_prototype_decoder = MLP(
            config.shared_dim,
            config.shared_dim,
            config.shared_dim,
            depth=2,
            dropout=config.dropout,
        )
```

### Script: `perturb_jepa/models/bridge.py`

This is the RNA/image condition readout path in `forward`. The current protected model chooses `rna_condition_readout="raw_linear_pseudobulk"` and `image_condition_readout="raw_linear_pooled"`, so the PLS-installed linear layers become the retrieval/shared geometry path.

```python
rna = self.rna_encoder(flat_gene_ids, flat_expression_values, token_mask=flat_token_mask)
rna_instance_shared = self.rna_projection(rna.cell_embedding).reshape(*rna_bag_shape, -1)
rna_aggregated = self.rna_bag_aggregator(rna_instance_shared, mask=rna_bag_mask)
rna_shared = rna_aggregated.bag_embedding
rna_pseudobulk_shared = self._rna_pseudobulk_shared(gene_ids, expression_values)
rna_raw_pseudobulk_shared = self._rna_raw_pseudobulk_shared(expression_values)
rna_raw_linear_shared = self._rna_raw_linear_pseudobulk_shared(expression_values)
rna_distilled_linear_shared = self._rna_distilled_linear_pseudobulk_shared(expression_values)
rna_distilled_residual_shared = rna_distilled_linear_shared + (
    self.rna_distilled_residual_scale * rna_raw_pseudobulk_shared
)
rna_condition_for_counterfactual = self._rna_condition_values(expression_values)
if self.config.rna_condition_readout == "encoder":
    pass
elif self.config.rna_condition_readout == "pseudobulk":
    rna_shared = rna_pseudobulk_shared
elif self.config.rna_condition_readout == "encoder_plus_pseudobulk":
    rna_shared = F.normalize(rna_shared + rna_pseudobulk_shared, dim=-1)
elif self.config.rna_condition_readout == "raw_pseudobulk":
    rna_shared = rna_raw_pseudobulk_shared
elif self.config.rna_condition_readout == "encoder_plus_raw_pseudobulk":
    rna_shared = F.normalize(rna_shared + rna_raw_pseudobulk_shared, dim=-1)
elif self.config.rna_condition_readout == "raw_linear_pseudobulk":
    rna_shared = rna_raw_linear_shared
elif self.config.rna_condition_readout == "encoder_plus_raw_linear_pseudobulk":
    rna_shared = F.normalize(rna_shared + rna_raw_linear_shared, dim=-1)
else:
    raise ValueError(f"unsupported rna_condition_readout: {self.config.rna_condition_readout}")
```

```python
image = self.image_encoder(flat_images, patch_mask=flat_patch_mask)
image_instance_shared = self.image_projection(image.image_embedding).reshape(*image_bag_shape, -1)
image_aggregated = self.image_bag_aggregator(image_instance_shared, mask=image_bag_mask)
image_shared = image_aggregated.bag_embedding
image_raw_shared = self._image_raw_shared(images)
image_raw_linear_shared = self._image_raw_linear_shared(images)
image_distilled_linear_shared = self._image_distilled_linear_shared(images)
image_distilled_residual_shared = image_distilled_linear_shared + (
    self.image_distilled_residual_scale * image_raw_shared
)
if self.config.image_condition_readout == "encoder":
    pass
elif self.config.image_condition_readout == "raw_pooled":
    image_shared = image_raw_shared
elif self.config.image_condition_readout == "encoder_plus_raw_pooled":
    image_shared = F.normalize(image_shared + image_raw_shared, dim=-1)
elif self.config.image_condition_readout == "raw_linear_pooled":
    image_shared = image_raw_linear_shared
elif self.config.image_condition_readout == "encoder_plus_raw_linear_pooled":
    image_shared = F.normalize(image_shared + image_raw_linear_shared, dim=-1)
else:
    raise ValueError(f"unsupported image_condition_readout: {self.config.image_condition_readout}")
```

### Script: `perturb_jepa/models/bridge.py`

This is the start-to-end counterfactual path inside `forward`.

```python
if shared_for_state is not None:
    z_state = self.state_head(shared_for_state)
    z_response = self.response_head(shared_for_state)
    delta, delta_gate, delta_base = self.predict_delta(z_state, perturbation)
    z_counterfactual = z_state + delta
    reverse_delta, reverse_gate, reverse_base = self.predict_delta(z_counterfactual, -perturbation)
    z_cycle = z_counterfactual + reverse_delta
    counterfactual_rna_delta, counterfactual_rna_aux = self._counterfactual_rna_delta(
        z_counterfactual,
        rna_condition_for_counterfactual,
        perturbation_id=perturbation_id,
        cell_line_id=cell_line_id,
        dose=dose,
        time=time,
    )
    counterfactual_rna = counterfactual_rna_delta
    if self.config.counterfactual_rna_residual and rna_condition_for_counterfactual is not None:
        counterfactual_rna = _match_last_dim(rna_condition_for_counterfactual, counterfactual_rna_delta.shape[-1])
        counterfactual_rna = counterfactual_rna + counterfactual_rna_delta
    outputs.update(
        {
            "z_state": z_state,
            "z_response": z_response,
            "z_counterfactual": z_counterfactual,
            "counterfactual_delta": delta,
            "counterfactual_gate": delta_gate,
            "counterfactual_base_delta": delta_base,
            "cycle_reconstruction": z_cycle,
            "cycle_delta": reverse_delta,
            "cycle_gate": reverse_gate,
            "cycle_base_delta": reverse_base,
            "counterfactual_rna_delta": counterfactual_rna_delta,
            "counterfactual_rna": counterfactual_rna,
            "counterfactual_image": self.image_prototype_decoder(z_counterfactual),
            **counterfactual_rna_aux,
        }
    )
```

### Script: `perturb_jepa/models/bridge.py`

This is the program-factorized counterfactual decoder implementation that was tested in the later loop.

```python
def _counterfactual_rna_delta(
    self,
    z_counterfactual: torch.Tensor,
    source_rna: torch.Tensor | None,
    perturbation_id: torch.Tensor,
    cell_line_id: torch.Tensor,
    dose: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not self.config.counterfactual_rna_program_factorized:
        return self.rna_distribution_decoder(z_counterfactual), {}
    assignment = self.counterfactual_rna_program_assignment
    decoder_input = z_counterfactual
    source_program_context = torch.zeros(
        z_counterfactual.shape[0],
        self.config.counterfactual_rna_num_programs,
        device=z_counterfactual.device,
        dtype=z_counterfactual.dtype,
    )
    if self.config.counterfactual_rna_program_conditioned:
        if source_rna is not None:
            source_program_context = _program_means(
                _match_last_dim(source_rna, assignment.numel()),
                assignment,
                num_programs=self.config.counterfactual_rna_num_programs,
            )
        decoder_input = torch.cat((z_counterfactual, source_program_context), dim=-1)
    metadata_context = torch.zeros(
        z_counterfactual.shape[0],
        self.config.perturbation.num_perturbations + self.config.perturbation.num_cell_lines + 2,
        device=z_counterfactual.device,
        dtype=z_counterfactual.dtype,
    )
    if self.config.counterfactual_rna_program_metadata_context:
        metadata_context = torch.cat(
            (
                F.one_hot(perturbation_id, num_classes=self.config.perturbation.num_perturbations).to(
                    dtype=z_counterfactual.dtype
                ),
                F.one_hot(cell_line_id, num_classes=self.config.perturbation.num_cell_lines).to(
                    dtype=z_counterfactual.dtype
                ),
                dose.to(dtype=z_counterfactual.dtype).unsqueeze(-1),
                time.to(dtype=z_counterfactual.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        decoder_input = torch.cat((decoder_input, metadata_context), dim=-1)
    program_delta = self.counterfactual_program_decoder(decoder_input)
    program_gene_delta = program_delta.index_select(dim=1, index=assignment)
    within_program_residual = torch.zeros_like(program_gene_delta)
    if self.config.counterfactual_rna_within_program_residual:
        raw_residual = _match_last_dim(self.rna_distribution_decoder(z_counterfactual), program_gene_delta.shape[-1])
        within_program_residual = raw_residual - _program_gene_means(raw_residual, assignment)
    delta = program_gene_delta + within_program_residual
    return delta, {
        "counterfactual_rna_program_delta": program_delta,
        "counterfactual_rna_program_gene_delta": program_gene_delta,
        "counterfactual_rna_within_program_residual": within_program_residual,
        "counterfactual_rna_source_program_context": source_program_context,
        "counterfactual_rna_metadata_context": metadata_context,
        "counterfactual_rna_program_decoder_input": decoder_input,
    }
```

### Script: `scripts/evaluate_prefit_pls_readout.py`

This is the protected PLS architecture installation path: fit PLS on train condition arrays, build the bridge, install PLS into raw-linear readout heads, then evaluate with Step 0 metrics.

```python
experiment_config = _experiment_config_for_dataset(
    dataset,
    steps=0,
    device=args.device,
    model_dim=max(4, args.dim),
    shared_dim=args.dim,
    dropout=0.0,
    rna_condition_readout="raw_linear_pseudobulk",
    rna_pseudobulk_normalize=False,
    image_condition_readout="raw_linear_pooled",
    image_raw_normalize=False,
    bag_aggregator="mean",
    num_bag_prototypes=1,
    rna_mask_weight=0.0,
    image_mask_weight=0.0,
    jepa_weight=0.0,
    align_weight=0.0,
    mmd_weight=0.0,
    sliced_wasserstein_weight=0.0,
    perturbation_cls_weight=0.0,
    batch_adv_weight=0.0,
    counterfactual_weight=0.0,
    cycle_weight=0.0,
    response_bottleneck_weight=0.0,
    shared_variance_weight=0.0,
    shared_covariance_weight=0.0,
)
model = experiment_config.build_model().to(args.device)
install_prefit_pls_readout(model, readout, freeze=True, device=args.device)

metrics = evaluate_step0(
    dataset,
    model,
    split="test",
    train_split="train",
    device=args.device,
    bag_size=dataset.config.cells_per_condition,
    seed=args.seed,
    label_shuffle_repeats=20,
)
```

### Script: `scripts/train_pls_distilled_head.py`

This is the trainable clone handoff path: build the same protected bridge, install frozen PLS retrieval heads, install trainable distilled heads from the same PLS weights, freeze everything else, then train only the student heads.

```python
dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
bag_size = dataset.config.cells_per_condition
train_arrays = _condition_arrays(dataset, "train")
readout = fit_pls_readout(train_arrays["rna_mean"], train_arrays["image_mean"], rank=args.rank)

experiment_config = _experiment_config_for_dataset(
    dataset,
    steps=args.steps,
    device=args.device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    dropout=0.0,
    model_dim=max(4, args.rank),
    shared_dim=args.rank,
    rna_condition_readout="raw_linear_pseudobulk",
    rna_pseudobulk_normalize=False,
    image_condition_readout="raw_linear_pooled",
    image_raw_normalize=False,
    bag_aggregator="mean",
    num_bag_prototypes=1,
    rna_mask_weight=0.0,
    image_mask_weight=0.0,
    jepa_weight=0.0,
    align_weight=0.0,
    mmd_weight=0.0,
    sliced_wasserstein_weight=0.0,
    perturbation_cls_weight=0.0,
    batch_adv_weight=0.0,
    counterfactual_weight=0.0,
    cycle_weight=0.0,
    response_bottleneck_weight=0.0,
    shared_variance_weight=0.0,
    shared_covariance_weight=0.0,
)
model = experiment_config.build_model().to(args.device)
install_prefit_pls_readout(model, readout, freeze=True, device=args.device)
if args.student_head in {"linear_clone", "residual_mlp"}:
    install_prefit_pls_distillation_head(model, readout, freeze=False, device=args.device)
_freeze_all_parameters(model)
_unfreeze_distilled_heads(model, args.student_head)
```

### Script: `scripts/train_clone_counterfactual_decoder.py`

This is the clone-frozen counterfactual training path: keep protected PLS and linear clone heads frozen, optionally enable residual/program-factorized RNA decoding, then train only the counterfactual modules.

```python
experiment_config = _experiment_config_for_dataset(
    dataset,
    steps=args.steps,
    device=args.device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    dropout=0.0,
    model_dim=max(4, args.rank),
    shared_dim=args.rank,
    rna_condition_readout="raw_linear_pseudobulk",
    rna_pseudobulk_normalize=False,
    image_condition_readout="raw_linear_pooled",
    image_raw_normalize=False,
    bag_aggregator="mean",
    num_bag_prototypes=1,
    rna_mask_weight=0.0,
    image_mask_weight=0.0,
    jepa_weight=0.0,
    align_weight=0.0,
    mmd_weight=0.0,
    sliced_wasserstein_weight=0.0,
    perturbation_cls_weight=0.0,
    batch_adv_weight=0.0,
    counterfactual_weight=0.0,
    cycle_weight=0.0,
    response_bottleneck_weight=0.0,
    shared_variance_weight=0.0,
    shared_covariance_weight=0.0,
    counterfactual_rna_residual=args.residual_rna,
    counterfactual_rna_program_factorized=args.program_factorized_rna,
    counterfactual_rna_num_programs=dataset.config.num_programs if args.program_factorized_rna else 0,
    counterfactual_rna_program_assignment=tuple(int(value) for value in dataset.gene_program_assignment)
    if args.program_factorized_rna
    else (),
    counterfactual_rna_within_program_residual=args.within_program_residual,
    counterfactual_rna_program_conditioned=args.program_condition_source,
    counterfactual_rna_program_metadata_context=args.program_metadata_context,
    counterfactual_rna_program_decoder_depth=1 if args.linear_program_decoder else 2,
)
model = experiment_config.build_model().to(args.device)
install_prefit_pls_readout(model, readout, freeze=True, device=args.device)
install_prefit_pls_distillation_head(model, readout, freeze=True, device=args.device)
if args.program_factorized_rna:
    _zero_init_program_factorized_decoder(model, zero_within=args.within_program_residual)
elif args.residual_rna:
    _zero_init_rna_delta_decoder(model)
_freeze_all_parameters(model)
_unfreeze_counterfactual_modules(
    model,
    train_perturbation_encoder=args.train_perturbation_encoder,
    program_factorized=args.program_factorized_rna,
    within_program_residual=args.within_program_residual,
)
```

## Actual Best Reference Code

### Script: `perturb_jepa/training/prefit_readout.py`

Current protected shared-geometry readout: closed-form low-rank PLS maps from train condition RNA/image means into a shared space.

```python
@dataclass(frozen=True)
class LinearReadoutMap:
    weight: np.ndarray
    bias: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) @ self.weight + self.bias


@dataclass(frozen=True)
class PrefitPLSReadout:
    rna: LinearReadoutMap
    image: LinearReadoutMap
    rank: int
    output_standardize: bool = False
    singular_values: np.ndarray | None = None


def fit_pls_readout(
    rna_values: np.ndarray,
    image_values: np.ndarray,
    *,
    rank: int,
    output_standardize: bool = False,
) -> PrefitPLSReadout:
    rna = np.asarray(rna_values, dtype=np.float32)
    image = np.asarray(image_values, dtype=np.float32)
    x_mean, x_std, xz = _standardize_fit_transform(rna)
    y_mean, y_std, yz = _standardize_fit_transform(image)
    cross = xz.T @ yz / max(1, xz.shape[0] - 1)
    u, singular, vt = np.linalg.svd(cross, full_matrices=False)
    keep = min(int(rank), u.shape[1], vt.shape[0])
    x_projection = u[:, :keep] * np.sqrt(singular[:keep])[None, :]
    y_projection = vt.T[:, :keep] * np.sqrt(singular[:keep])[None, :]
    return PrefitPLSReadout(
        rna=_compose_map(x_mean, x_std, x_projection, np.zeros(keep), np.ones(keep)),
        image=_compose_map(y_mean, y_std, y_projection, np.zeros(keep), np.ones(keep)),
        rank=keep,
        output_standardize=output_standardize,
        singular_values=singular[:keep],
    )


def install_prefit_pls_readout(
    model: torch.nn.Module,
    readout: PrefitPLSReadout,
    *,
    freeze: bool = True,
    device: torch.device | str | None = None,
) -> None:
    _install_linear_map(model.rna_raw_linear_projection, readout.rna, freeze=freeze, device=device)
    _install_linear_map(model.image_raw_linear_projection, readout.image, freeze=freeze, device=device)
```

### Script: `scripts/run_family_m_transport_baselines.py`

The no-batch biological key and matching references used as comparators for Families N and O.

```python
BIOLOGICAL_KEY_FIELDS = ("perturbation_id", "cell_line_id", "dose", "time")


def biological_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in BIOLOGICAL_KEY_FIELDS)


def pair_records(dataset: SyntheticBiologyLiteDataset, *, split: str) -> list[dict[str, Any]]:
    records = []
    for pair in _counterfactual_pairs(dataset, split=split):
        control_cells = np.asarray(dataset.expression_values[pair["control_group"]], dtype=float)
        target_cells = np.asarray(dataset.expression_values[pair["target_group"]], dtype=float)
        metadata = {key: value for key, value in pair.items() if not key.endswith("_group")}
        records.append(
            {
                **metadata,
                "biological_key": biological_key(metadata),
                "control_cells": control_cells,
                "target_cells": target_cells,
                "control_mean": control_cells.mean(axis=0),
                "target_mean": target_cells.mean(axis=0),
            }
        )
    return records


def predict_matched_perturbed_mean(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> tuple[list[np.ndarray], dict[str, float]]:
    target_by_key = group_mean_by_key(train_records, value_name="target_mean")
    global_target = np.stack([record["target_mean"] for record in train_records]).mean(axis=0)
    predictions = []
    exact_hits = 0
    for record in test_records:
        value = target_by_key.get(record["biological_key"])
        if value is None:
            value = global_target
        else:
            exact_hits += 1
        predictions.append(value)
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "batch_id_excluded": 1.0,
    }
```

### Script: `scripts/run_family_n_distillation.py`

Current best expression-space counterfactual reference: train-only biological-key condition mean table.

```python
class TrainOnlyConditionalMeanTable:
    """Train-split-only target mean lookup keyed by biological condition fields."""

    def __init__(self, train_records: list[dict[str, Any]], *, value_name: str = "target_mean") -> None:
        self.value_name = value_name
        self.train_record_count = len(train_records)
        grouped: dict[tuple[Any, ...], list[np.ndarray]] = {}
        perturbation_grouped: dict[Any, list[np.ndarray]] = {}
        for record in train_records:
            key = biological_key(record)
            value = np.asarray(record[value_name], dtype=float)
            grouped.setdefault(key, []).append(value)
            perturbation_grouped.setdefault(record["perturbation_id"], []).append(value)
        self.target_by_key = {key: np.stack(values).mean(axis=0) for key, values in grouped.items()}
        self.perturbation_mean = {
            perturbation_id: np.stack(values).mean(axis=0)
            for perturbation_id, values in perturbation_grouped.items()
        }
        self.global_mean = np.stack([np.asarray(record[value_name], dtype=float) for record in train_records]).mean(axis=0)

    def predict(self, records: list[dict[str, Any]]) -> tuple[list[np.ndarray], dict[str, float]]:
        predictions = []
        counts = {"exact": 0, "nearest_same_perturbation_cell": 0, "global_perturbation_mean": 0, "global_train_mean": 0}
        for record in records:
            prediction, fallback = self._predict_one(record)
            counts[fallback] += 1
            predictions.append(prediction)
        rows = len(records)
        return predictions, {
            "rows": float(rows),
            "exact_match_fraction": float(counts["exact"] / max(1, rows)),
            "batch_id_excluded": 1.0,
            "fit_split_train_only": 1.0,
            "teacher_target_test_rows_used": 0.0,
        }

    def _predict_one(self, record: dict[str, Any]) -> tuple[np.ndarray, str]:
        exact = self.target_by_key.get(biological_key(record))
        if exact is not None:
            return exact, "exact"
        perturbation_value = self.perturbation_mean.get(record["perturbation_id"])
        if perturbation_value is not None:
            return perturbation_value, "global_perturbation_mean"
        return self.global_mean, "global_train_mean"
```

### Script: `scripts/run_family_n_distillation.py`

Learned expression-space students tested against the train-only teacher. These passed leakage diagnostics but did not beat the table.

```python
class ConditionFeatureEncoder:
    """No-batch biological condition and source-program feature encoder."""

    def transform(self, records: list[dict[str, Any]]) -> np.ndarray:
        rows = []
        perturb_index = {value: index for index, value in enumerate(self.perturbation_ids)}
        cell_index = {value: index for index, value in enumerate(self.cell_line_ids)}
        key_index = {value: index for index, value in enumerate(self.biological_keys)}
        for record in records:
            values: list[float] = [1.0]
            perturbation_one_hot = np.zeros(len(self.perturbation_ids), dtype=float)
            if int(record["perturbation_id"]) in perturb_index:
                perturbation_one_hot[perturb_index[int(record["perturbation_id"])]] = 1.0
            values.extend(perturbation_one_hot.tolist())
            cell_one_hot = np.zeros(len(self.cell_line_ids), dtype=float)
            if int(record["cell_line_id"]) in cell_index:
                cell_one_hot[cell_index[int(record["cell_line_id"])]] = 1.0
            values.extend(cell_one_hot.tolist())
            values.append((float(record["dose"]) - self.dose_mean) / self.dose_std)
            values.append((float(record["time"]) - self.time_mean) / self.time_std)
            key_one_hot = np.zeros(len(self.biological_keys), dtype=float)
            key = biological_key(record)
            if key in key_index:
                key_one_hot[key_index[key]] = 1.0
            values.extend(key_one_hot.tolist())
            source_program = program_means(record["control_mean"], self.dataset.gene_program_assignment)[0]
            values.extend(((source_program - self.source_program_mean) / self.source_program_std).tolist())
            rows.append(np.asarray(values, dtype=float))
        return np.stack(rows, axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "batch_id_feature_present": any("batch" in name for name in self.feature_names),
            "fit_split_train_only": True,
            "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        }


class LinearConditionalMeanModel:
    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, *, ridge_alpha: float) -> "LinearConditionalMeanModel":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        penalty = np.eye(x.shape[1], dtype=float) * float(ridge_alpha)
        penalty[0, 0] = 0.0
        system = x.T @ x + penalty
        rhs = x.T @ y
        weights = np.linalg.solve(system, rhs)
        prediction = x @ weights
        return cls(weights, ridge_alpha=ridge_alpha, train_mse=float(np.mean((prediction - y) ** 2)))
```

### Script: `scripts/run_family_o_count_likelihood.py`

Current count-aware audit and raw-count resolution. The synthetic generator exposes raw `observed_counts`, so pseudo-counts were not used in the Family O run.

```python
def resolve_count_matrix(
    dataset: SyntheticBiologyLiteDataset,
    *,
    pseudo_count_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if hasattr(dataset, "observed_counts") and dataset.observed_counts is not None:
        counts = np.asarray(dataset.observed_counts, dtype=float)
        path = "raw_synthetic_observed_counts"
        pseudo = False
        scale = None
    else:
        counts = pseudo_counts_from_expression(dataset.expression_values, pseudo_count_scale=pseudo_count_scale)
        path = "synthetic_pseudo_count_from_expression_values"
        pseudo = True
        scale = float(pseudo_count_scale)
    diagnostics = count_path_diagnostics(dataset, counts, count_path=path, pseudo_count_used=pseudo, pseudo_count_scale=scale)
    return counts, diagnostics


def count_pair_records(dataset: SyntheticBiologyLiteDataset, counts: np.ndarray, *, split: str) -> list[dict[str, Any]]:
    records = []
    for pair in _counterfactual_pairs(dataset, split=split):
        control_counts = np.asarray(counts[pair["control_group"]], dtype=float)
        target_counts = np.asarray(counts[pair["target_group"]], dtype=float)
        control_expression = np.asarray(dataset.expression_values[pair["control_group"]], dtype=float)
        target_expression = np.asarray(dataset.expression_values[pair["target_group"]], dtype=float)
        metadata = {key: value for key, value in pair.items() if not key.endswith("_group")}
        records.append(
            {
                **metadata,
                "biological_key": biological_key(metadata),
                "control_mean": control_expression.mean(axis=0),
                "target_mean": target_expression.mean(axis=0),
                "control_count_mean": control_counts.mean(axis=0),
                "target_count_mean": target_counts.mean(axis=0),
                "target_count_cells": target_counts,
            }
        )
    return records
```

### Script: `scripts/run_family_o_count_likelihood.py`

Count-aware likelihood scoring and the count MLP that was tested but not promoted.

```python
class CountMeanMLP(torch.nn.Module):
    """Small no-batch decoder that predicts per-gene log-count means."""

    def __init__(self, input_dim: int, genes: int, *, hidden_dim: int, initial_mean: np.ndarray) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, genes),
        )
        final = self.net[-1]
        torch.nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.log(torch.as_tensor(np.clip(initial_mean, 1e-4, None), dtype=torch.float32)))

    def forward(self, features: torch.Tensor, *, min_log_mean: float, max_log_mean: float) -> torch.Tensor:
        log_mu = torch.clamp(self.net(features), min=float(min_log_mean), max=float(max_log_mean))
        return torch.clamp(torch.exp(log_mu), min=1e-6, max=float(np.exp(max_log_mean)))


def poisson_nll_torch(y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    mu = torch.clamp(mu, min=1e-8)
    return mu - y * torch.log(mu) + torch.lgamma(y + 1.0)


def nb_nll_torch(y: torch.Tensor, mu: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
    mu = torch.clamp(mu, min=1e-8)
    alpha = torch.clamp(dispersion, min=1e-8)
    theta = 1.0 / alpha
    log_prob = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y * (torch.log(mu) - torch.log(theta + mu))
    )
    return -log_prob
```

## Source File Index

| role | script/module |
|---|---|
| Synthetic generator | `perturb_jepa/training/synthetic_biology_lite.py` |
| Step 0 baseline runner | `scripts/run_synthetic_lite_step0.py` |
| Protected PLS readout implementation | `perturb_jepa/training/prefit_readout.py` |
| PLS readout evaluator | `scripts/evaluate_prefit_pls_readout.py` |
| PLS trainable clone/distillation | `scripts/train_pls_distilled_head.py` |
| Counterfactual decoder training | `scripts/train_clone_counterfactual_decoder.py` |
| Family M matching/transport baselines | `scripts/run_family_m_transport_baselines.py` |
| Family N condition-mean distillation | `scripts/run_family_n_distillation.py` |
| Family O count-likelihood diagnostics | `scripts/run_family_o_count_likelihood.py` |
| Family M tests | `tests/test_family_m_transport_baselines.py` |
| Family N tests | `tests/test_family_n_distillation.py` |
| Family O tests | `tests/test_family_o_count_likelihood.py` |

## Decision

Do not promote a neural JEPA candidate from the current synthetic exact-key split. Keep the rank-3 PLS readout as protected synthetic geometry, keep Family N as the expression-space reference, and keep Family O as the count-aware reference. The next serious model benchmark should evaluate no-exact-key generalization before more architecture search.
