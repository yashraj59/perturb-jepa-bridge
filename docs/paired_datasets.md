# Paired and Weakly Paired Dataset Assessment

Pairing tiers used in this repository:

- Tier 1: true paired image-expression at cell, spot, tile, or region level.
- Tier 2: weak sample, well, organoid, condition, or compound-level pairing.
- Tier 3: optical pooled image-to-genotype/barcode pairing.
- Tier 4: unpaired but useful unimodal data.

## Candidate Datasets

| Dataset | Tier | Pairing | System | Perturbation | Dose/time/cell line | Access | License / commercial status | Size | Pairing key evidence | Use in this repo | Risks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SpatialLIBD | Tier 1 | image + spatial transcriptomics | human DLPFC tissue | none | sample/section, no perturbation | Bioconductor https://www.bioconductor.org/packages/release/data/experiment/html/spatialLIBD.html; DOI https://doi.org/10.1038/s41593-020-00787-0 | Artistic-2.0 package; raw terms `unclear` | 47,681 spots | spot-level spatial coordinates and image scale factors in `SpatialExperiment` | direct image-to-expression validation | not perturbational; spatial autocorrelation |
| 10x Genomics Visium public examples | Tier 1 | image + spatial transcriptomics | tissues vary | usually none | dataset-dependent | https://www.10xgenomics.com/resources/datasets | `unclear`; verify selected dataset terms | varies | spot barcodes, coordinates, histology image | true paired spot benchmark | selected dataset may lack reuse clarity |
| STimage-1K4M | Tier 1 | histology tile + expression spot/region | cancer/tissue spatial datasets | usually none | dataset-dependent | Hugging Face https://huggingface.co/datasets/jiawennnn/STimage-1K4M; PubMed https://pubmed.ncbi.nlm.nih.gov/38947920/; arXiv https://arxiv.org/abs/2406.06393 | license/commercial status `unclear`; verify Hugging Face dataset card before redistribution | 1,149 images and spot-level gene-expression pairs reported by source | spot/tile pairing reported in dataset metadata | optional paired benchmark | not verified for this repo until source metadata is staged |
| HER2 breast cancer spatial transcriptomics | Tier 1 | H&E image + expression spots | breast cancer tissue | none | sample-level | original spatial transcriptomics resources; verify exact portal before use | `unclear` | moderate | spot coordinates and histology images | optional true paired validation | disease tissue, not perturbation |
| PERISCOPE / Cell Painting Gallery cpg0021 | Tier 3 | image + perturbation barcode/guide | A549 / HeLa cells | CRISPR guides | cell line present; dose/time not primary | S3 `s3://cellpainting-gallery/cpg0021-periscope/`; registry https://registry.opendata.aws/cellpainting-gallery/; DOI https://doi.org/10.1038/s41592-024-02537-7 | Gallery CC0; article CC BY 4.0; `commercial_ok` | large | in situ sequencing barcode/guide assignment | image-to-guide/gene retrieval, held-out guide/gene validation | no expression readout |
| Feldman optical pooled screens / IDR0071 | Tier 3 | image + sgRNA barcode | cultured cells | CRISPR guides | experiment-dependent | IDR study https://idr.openmicroscopy.org/search/?query=idr0071; protocol DOI https://doi.org/10.1038/s41596-021-00653-8; original DOI https://doi.org/10.1016/j.cell.2019.09.016 | code MIT; dataset commercial status `unclear` | targeted screen | barcode/cell assignment from in situ sequencing | image encoder perturbation validation | smaller and method-specific |
| BF-MoA | Tier 2 or Tier 4 | brightfield + fluorescence images; compound/MoA labels | cultured cells | compounds | metadata includes compound/MoA and plate/well/site; dose/time if present in tables | Figshare https://doi.org/10.17044/SCILIFELAB.21378906 | CC BY 4.0, `commercial_ok` | metadata small, images very large | well/site image metadata, but no RNA pairing | image pretraining and MoA retrieval; weak compound-level bridge only | not expression paired |
| JUMP cpg0016 + external L1000 overlap | Tier 2 | compound/gene-level weak correspondence | human cells | compounds, ORF, CRISPR | source-specific | Cell Painting Gallery https://registry.opendata.aws/cellpainting-gallery/; JUMP docs https://jump-cellpainting.github.io/ | CC0/BSD-3-Clause, `commercial_ok` | multi-TB raw/profiles | shared compound/gene identifiers only unless a matched table is supplied | weak condition/compound bridge and image baselines | do not call paired without matched well/sample table |
| LINCS L1000 + Cell Painting overlap | Tier 2 | transcriptomic signatures + morphology for overlapping perturbagens | human cell lines | compounds/genetic perturbations | dose/time/cell line in LINCS; morphology varies | CLUE/LINCS https://clue.io/ and Cell Painting Gallery sources | CLUE terms require verification, `unclear` or `restricted` depending access | large | compound/cell-line/time metadata, not same-cell | optional weak compound-level validation | unrelated experiments can inflate claims |
| Sci-Plex3 | Tier 4 | RNA only | human cancer cell lines | compounds | dose/cell-line/control | GEO https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944; DOI https://doi.org/10.1126/science.aax6234 | harmonized scPerturb CC BY 4.0, `commercial_ok` | large | none to images | RNA benchmark | no image pairing |
| Replogle / Norman / Adamson / Dixit Perturb-seq | Tier 4 | RNA only | human/mouse cell systems | CRISPR | perturbation/control; dose usually not applicable | sources in `docs/public_datasets.md` | mixed: scPerturb CC BY 4.0 or GEO `unclear` | large | none to images | RNA pretraining/counterfactual validation | not image paired |
| RxRx1 | Tier 4 | image only | human cells | siRNA | cell type/siRNA | https://www.rxrx.ai/rxrx1 | non-commercial, `research_only` | 125k+ images | well/site image metadata only | batch stress test | non-commercial terms and no RNA |

## Use Policy

Verified fact: true paired public resources found here are spatial
transcriptomics plus images, not same-cell brightfield plus scRNA perturbation.
Optical pooled screens pair images to perturbation identity, not gene expression.

Implementation result: `scripts/build_paired_manifest.py` requires explicit
pairing columns for cell, spot, tile, well, or sample pairing. Without those
columns it writes condition-level weak pairing only.

Inference: condition-level overlap between Sci-Plex3 compounds and image MoA
datasets is useful for weak bridge stress tests, but it cannot validate
same-cell or same-well RNA-image learning.

Open uncertainty: STimage-1K4M and some LINCS/Cell Painting overlaps require
dataset-specific metadata staging before this repo can mark pairing and license
status as verified.
