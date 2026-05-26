# BioOperator-JEPA Phase 5 Research Journal


## BOJ000: Operator Floor Reproduction

**Hypothesis**: before neural operators, the Phase 4 train-only action-ridge floor must be exactly reproducible from cached teacher latents.

**Result**: eval ridge improvement `0.0057`, delta cosine `0.3980`, rank `10.2835`.

**Decision label**: `PHASE5_OPERATOR_FLOOR_REPRODUCED`.


## Stage B: Operator Contract Tests

**Result**: `pytest tests/test_biooperator_contracts.py tests/test_biooperator_losses.py` passed with `5 passed`. Sign, gradient, hinge/magnitude, and ridge-equivalence contracts are satisfied. BioOperator implementation is permitted.

## BOJ001: frozen_neural_ridge

**Hypothesis**: neural/operator form can reproduce or preserve the train-only action-ridge transition floor without sign, magnitude, or leakage failure.

**Implementation**: mode `frozen_neural_ridge`, steps `100`, rank `8`, device `cuda`.

**Tier result**: eval transition improvement `0.0057`, eval delta cosine `0.3980`, eval recall@1 `0.4815`, eval delta rank `10.2835`, magnitude ratio `0.7744`.

**Decision label**: `TIER1_KEEP_OPERATOR_MATCHES_FLOOR`.


## BOJ002: frozen_control_affine

**Hypothesis**: neural/operator form can reproduce or preserve the train-only action-ridge transition floor without sign, magnitude, or leakage failure.

**Implementation**: mode `frozen_control_affine`, steps `100`, rank `8`, device `cuda`.

**Tier result**: eval transition improvement `0.0025`, eval delta cosine `0.3603`, eval recall@1 `0.3704`, eval delta rank `7.5812`, magnitude ratio `0.7579`.

**Decision label**: `TIER1_DISCARD_OPERATOR_BELOW_FLOOR`.

