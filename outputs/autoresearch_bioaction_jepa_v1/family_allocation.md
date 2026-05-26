# Family Allocation

| Family | Experiments used | Tier 1 keeps | Tier 2 passes | Tier 3 wins | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| A Minimal Real BioAction-JEPA | 3 | 0 | 0 | 0 | failed Tier 1 due batch leakage / weak RNA->image retrieval |
| B Missing-Modality Binding | 0 | 0 | 0 | 0 | pending |
| C Biological Action World Model | 0 | 0 | 0 | 0 | pending |
| D Biological Program/Graph Priors | 0 | 0 | 0 | 0 | pending |
| E Distributional Cell-State JEPA | 0 | 0 | 0 | 0 | pending |
| F Batch-Invariant Latent JEPA | 2 | 0 | 0 | 0 | failed; leakage persisted |

Hard experiment cap from prompt: 20 unless explicitly changed later.

Autonomous loop status: stopped. Stop condition fired because batch leakage dominated the latent state across two candidate families, A and F.
