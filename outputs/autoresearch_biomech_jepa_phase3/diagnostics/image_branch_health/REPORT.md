# Image Branch Health Audit

Decision label: `IMAGE_BRANCH_AUDIT_HEALTHY`

## Key Metrics

- Trained online image->RNA recall@1: `0.0000`
- Trained teacher image->RNA recall@1: `0.1250`
- Trained online RNA->image recall@1: `0.1875`
- Random online image->RNA recall@1: `0.1875`
- Image teacher `z_bio` effective rank: `5.9059`
- Image teacher `z_tech` effective rank: `3.6370`
- Image region target variance: `0.000911`
- Image/RNA branch gradient norm ratio: `0.9884`
- Cross-modal weighted/total loss ratio: `0.3436`

## Interpretation

This audit distinguishes loader/data failure, latent collapse, and loss/gradient imbalance before any Phase 3 architecture change.
