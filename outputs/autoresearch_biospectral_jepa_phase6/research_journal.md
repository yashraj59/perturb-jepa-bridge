# BioSpectral-JEPA Phase 6 Research Journal


## BSJ000: Rank Bottleneck Audit

**Result**: full ridge eval improvement `0.0057`, low-rank eval improvement `0.0046`, neural low-rank equivalence pass `1.0`.

**Decision label**: `PHASE6_REOPEN_BIOSPECTRAL_APPROVED`.


## BSJ001: frozen_neural_low_rank_equivalence

**Implementation**: mode `frozen_neural_low_rank_equivalence`, seed `0`, device `cpu`, steps `80`.

**Result**: eval transition improvement `0.0046`, delta cosine `0.3877`, recall@1 `0.4074`, delta rank `6.7681`, magnitude ratio `0.7585`, floor gap `-0.0011`.

**Decision label**: `BSJ001_KEEP_NEURAL_LOWRANK_MATCHES_ANALYTIC`.


## BSJ002: frozen_full_ridge_floor_wrapper_zero_residual

**Implementation**: mode `frozen_full_ridge_floor_wrapper_zero_residual`, seed `0`, device `cpu`, steps `80`.

**Result**: eval transition improvement `0.0057`, delta cosine `0.3980`, recall@1 `0.4815`, delta rank `10.2835`, magnitude ratio `0.7744`, floor gap `-0.0000`.

**Decision label**: `BSJ002_KEEP_FLOOR_WRAPPER_MATCHES_FULL_RIDGE`.


## BSJ003: frozen_rank_ladder_router

**Implementation**: mode `frozen_rank_ladder_router`, seed `0`, device `cpu`, steps `80`.

**Result**: eval transition improvement `0.0057`, delta cosine `0.3980`, recall@1 `0.4815`, delta rank `10.2835`, magnitude ratio `0.7744`, floor gap `-0.0000`.

**Decision label**: `BSJ003_KEEP_RANK_LADDER_PRESERVES_FLOOR`.


## BSJ004: frozen_floor_plus_spectral_residual

**Implementation**: mode `frozen_floor_plus_spectral_residual`, seed `0`, device `cpu`, steps `80`.

**Result**: eval transition improvement `0.0050`, delta cosine `0.4059`, recall@1 `0.4074`, delta rank `10.4574`, magnitude ratio `0.8000`, floor gap `-0.0007`.

**Decision label**: `BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR`.

