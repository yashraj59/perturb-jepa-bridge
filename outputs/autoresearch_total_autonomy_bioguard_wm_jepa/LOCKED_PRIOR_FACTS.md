# Locked Prior Facts

Evidence files checked:
- `outputs/autoresearch_bioaction_jepa_v1/final_report.md`: `True`
- `outputs/autoresearch_biotech_jepa_phase2/final_report.md`: `True`
- `outputs/autoresearch_biomech_jepa_phase3/final_report.md`: `True`
- `outputs/autoresearch_bioflow_jepa_phase4/final_report.md`: `True`
- `outputs/autoresearch_biooperator_jepa_phase5/final_report.md`: `True`
- `outputs/autoresearch_biospectral_jepa_phase6/final_report.md`: `True`
- `outputs/autoresearch_bioguard_jepa_phase7/final_report.md`: `True`
- `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_OPERATOR_AUDIT.md`: `True`
- `outputs/autoresearch_biotech_jepa_phase2/BIOTECH_JEPA_CODE_INDEX.md`: `True`
- `outputs/autoresearch_biotech_jepa_phase2/NORMAN_CONTEXT_AUDIT.md`: `True`
- `outputs/autoresearch_synth_lite/FULL_ARCHITECTURE_CODE_BUNDLE.md`: `True`

Phase 1 BioAction-JEPA:
- Real JEPA identity was implemented.
- Stop reason: batch leakage dominated latent state.
- Protected PLS remained model of record.

Phase 2 BioTech-JEPA:
- z_bio / z_tech separation was implemented.
- Synthetic genetic-anchor audit reopened architecture search.
- Norman is RNA-only in processed h5ad; batch and chemical dose cannot be validated there.

Phase 3 BioMechanistic-JEPA:
- Delta targets had headroom.
- BMJ001 delta operator anti-aligned / no useful signal.

Phase 4 BioFlow-JEPA:
- Action-ridge simple baseline was positive.
- Neural vector-field candidate failed with negative signal.

Phase 5 BioOperator-JEPA:
- Neural ridge-equivalence reproduced the train-only action-ridge floor.
- Low-rank control-affine operator fell below floor.

Phase 6 BioSpectral-JEPA:
- Floor wrapper and rank ladder preserved floor.
- Spectral residual improved train fit but fell below floor on held-out transition/retrieval.

Phase 7 BioGuard-JEPA:
- Decision: `PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA`.
- No residual passed train-only cross-fitted selection: `True`.
- Spectral, kernel, and program residuals all deployed zero-residual floor fallback.
- Leakage audit passed.
- No full BioGuard-JEPA candidate was trained.

Protected full-ridge transition floor:
- transition_source_cosine_improvement = 0.0057
- selected_delta_cosine = 0.3980
- transition_to_target_recall@1 = 0.4815
- delta_prediction_effective_rank = 10.2835
- delta_magnitude_ratio = 0.7744
