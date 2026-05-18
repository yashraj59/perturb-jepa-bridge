# Perturb-JEPA Bridge Real-Data Run Report

## Data acquisition log

| File | Size | SHA256 |
|---|---:|---|
| `data/raw/SrivatsanTrapnell2020_sciplex3.h5ad` | 2,526,631,614 bytes | `603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a` |
| `data/raw/bf_moa_data_tables.tar.gz` | 4,631,905 bytes | `4149c7a737806b74d6ed3c9e971f68a388367a535072247d727c43b52920854f` |
| `data/raw/bf_moa_figshare.json` | 19,066 bytes | `fc7013e058f76efaac77965c7b7203bff196cc1ef960e3d45737fbf8f8b49e9c` |
| `data/raw/P015080.tar.gz` | 4,607,761,887 bytes | `7d38b30c66cf668208cc8896b61e6a2065f82a90368c066d15cfe5259d770b9f` |

Extracted image plate: `data/raw/bf_moa_images/P015080`, 605 files, 5.1 GB.

sci-Plex3 verification: `799,317` cells x `110,983` genes. Observed metadata columns include `cell_line`, `time`, `dose_value`, `pathway_level_1`, `pathway_level_2`, `perturbation`, `target`, `pathway`, and `perturbation_type`.

BF-MoA Figshare plate count summary from `data/processed/bf_moa_manifest_raw.csv`:

| Plate | Manifest rows | Compounds | MoAs |
|---|---:|---:|---:|
| P015080 | 330 | 2 | 2 |
| P015081 | 330 | 2 | 2 |
| P015082 | 420 | 3 | 2 |
| P015083 | 420 | 3 | 2 |
| P015076 | 600 | 5 | 4 |
| P015077 | 600 | 5 | 4 |
| P015092 | 1,680 | 17 | 2 |
| P015093 | 1,680 | 17 | 2 |
| P015096 | 2,040 | 21 | 2 |
| P015097 | 2,040 | 21 | 2 |
| P015094 | 2,130 | 22 | 2 |
| P015095 | 2,130 | 22 | 2 |
| P015084 | 3,750 | 39 | 3 |
| P015085 | 3,750 | 39 | 3 |
| P015090 | 5,910 | 63 | 4 |
| P015091 | 5,910 | 63 | 4 |
| P015098 | 6,270 | 68 | 5 |
| P015099 | 6,270 | 68 | 5 |

## Compound intersection

The required first run downloaded the smallest BF-MoA brightfield plate, `P015080`. This plate contains 55 complete six-channel sites but only two compound labels:

| BF-MoA compound | MoA | Manifest rows | Sites |
|---|---|---:|---:|
| `CBK200949` | topoisomerase inhibitor | 90 | 5 |
| `dmso` | dmso | 240 | 5 |

After sci-Plex3 filtering to A549, 10uM, and 72h, and after compound normalization plus parenthetical synonym parsing, the shared compound set was empty.

Shared compounds: none.

RNA-side compounds after the selected filter included HDAC and kinase inhibitors such as `abexinostat`, `panobinostat`, `quisinostat`, `trichostatin a`, `tacedinaline`, `ruxolitinib`, and `tozasertib`. Image-side compounds were only `cbk200949` and `dmso`.

Because the hard requirement is at least six shared compounds, preprocessing halted before writing aligned bridge inputs. This is a data-design failure of the mandated one-smallest-plate subset, not a model-training result.

## Configuration

Created `configs/real_sciplex3_bfmoa.yaml` with:

| Setting | Value |
|---|---|
| Bridge key | `condition_key_medium` (`perturbation|dose|time`) |
| RNA input | `data/processed/sciplex3_a549_10uM_72h.h5ad` |
| Image input | `data/processed/bf_moa_manifest_aligned.csv` |
| Image channels | 6 |
| Image size | 224 |
| Split strategy | `heldout_perturbation` |
| Device | `cuda` |

The aligned RNA/image files were not created because `scripts/prepare_real_pairing.py` correctly halted on zero shared compounds.

## Training curves

Training was not run. The bridge contract requires overlapping condition IDs, and the mission requires at least six shared compounds. Running RNA pretrain, image pretrain, or bridge training after a failed intersection would produce invalid artifacts for this benchmark.

| Stage | Status | Reason |
|---|---|---|
| RNA pretrain | not run | aligned RNA subset unavailable after intersection failure |
| Image pretrain | not run | aligned BF-MoA manifest unavailable after intersection failure |
| Bridge | not run | zero shared compounds; `PairedConditionBagDataset` would be invalid |
| Counterfactual | not run | downstream benchmark halted before model training |

## Retrieval metrics

`metrics/retrieval_test.csv` was not generated because `checkpoints/bridge_real.pt` was not trained.

## Centroid + label-shuffle

`metrics/centroid_baselines_test.csv` was not generated because no learned test embeddings were produced.

## Counterfactual

`metrics/counterfactual.csv` was not generated because the run halted before downstream model training.

## Expression baselines

`metrics/expression_baselines.csv` was not generated because counterfactual evaluation arrays were not produced.

## Acceptance check

| Criterion | Status | Evidence |
|---|---|---|
| Learned beats metadata-only on same-MoA enrichment | fail | no learned model; no aligned condition set |
| Learned beats batch-only | fail | no learned model; no aligned condition set |
| Learned beats label-shuffle control | fail | no learned embeddings; no centroid baseline |
| Held-out perturbation degradation bounded | fail | no counterfactual evaluation |
| Batch adversarial pressure is real | fail | no bridge training log |
| At least six shared compounds | fail | shared set is empty for `P015080` and A549 10uM/72h sci-Plex3 |

The result is invalid for publication claims and must not be presented as a biology win.

## Limitations

- The requested smallest single-plate BF-MoA subset cannot satisfy the `>=6` compound-overlap requirement; `P015080` has only `CBK200949` and `dmso`.
- Among BF-MoA plates, every plate under roughly 10 GB has fewer than six compounds in the manifest; the first plates with six or more compounds are larger archives.
- The planned biological bridge remains condition-level and unpaired; no cell-to-image pairs were created.
- The intended time mismatch remains unresolved: sci-Plex3 72h is the closest documented bucket to BF-MoA 48h and would need to be reported in any successful run.
- The A549 versus U2OS cell-line difference remains a core limitation; the bridge key must drop `cell_line`, while retaining source cell line only as metadata.

## Citations

- Srivatsan et al., "Massively multiplex chemical transcriptomics at single-cell resolution." *Science* (2019). DOI: 10.1126/science.aax6234.
- Harrison et al., "Evaluating the utility of brightfield image data for mechanism of action prediction." *PLoS Computational Biology* (2023). DOI: 10.1371/journal.pcbi.1011323.
- Peidli et al., "scPerturb: harmonized single-cell perturbation data." *Nature Methods* (2024). DOI: 10.1038/s41592-023-02144-y.
- Replogle et al., "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq." *Cell* (2022). DOI: 10.1016/j.cell.2022.05.013. Context only; not used as bridge data.
