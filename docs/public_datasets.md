# Public Datasets for Perturb-JEPA Bridge

This catalog is evidence-grounded for benchmark setup. It does not claim pairing
unless source metadata exposes an explicit cell, spot, tile, well, sample, or
condition key.

## Dataset Table

| Dataset | Modality | Organism / system | Perturbation | Labels available | Dose / time / cell line | Technical metadata | Access | License / commercial status | Size / raw data | Useful stages |
|---|---|---|---|---|---|---|---|---|---|---|
| Sci-Plex3 / Srivatsan-Trapnell | scRNA-seq | human cancer cell lines | compounds | perturbation, controls | dose and cell line verified; time depends on metadata extract | batch/replicate/hash metadata | GEO `GSE139944`: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944; paper DOI https://doi.org/10.1126/science.aax6234; harmonized scPerturb Zenodo record https://doi.org/10.5281/zenodo.7041849 | GEO public but explicit reuse license not verified; scPerturb record is CC BY 4.0, `commercial_ok` with attribution for harmonized files | raw GEO tar and processed h5ad; large | RNA pretraining, counterfactual validation, biological validation |
| Replogle / Weissman 2022 | scRNA-seq | human K562 and RPE1 | CRISPRi | guide/gene perturbation, controls | cell population verified; dose/time generally not applicable | batch/library metadata in AnnData | Figshare+ DOI https://doi.org/10.25452/figshare.plus.20029387; paper DOI https://doi.org/10.1016/j.cell.2022.05.013; scPerturb https://doi.org/10.5281/zenodo.7041849 | Figshare+/scPerturb CC BY 4.0, `commercial_ok` with attribution | processed AnnData and pseudobulk; large | RNA pretraining, held-out perturbation testing |
| Norman / Weissman 2019 | scRNA-seq | human K562 | CRISPRa single/combinatorial | guide/gene perturbation, controls | cell line verified; dose/time not applicable | sample/library metadata | GEO `GSE133344`: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344; paper DOI https://doi.org/10.1126/science.aax4438; scPerturb https://doi.org/10.5281/zenodo.7041849 | GEO explicit license not verified; scPerturb CC BY 4.0, `commercial_ok` for harmonized file | raw matrix plus h5ad; large | RNA pretraining, combinatorial generalization |
| Adamson / Weissman 2016 | scRNA-seq | human cell models | CRISPRi | guide/gene perturbation, controls | dose/time not applicable | batch/library metadata may be present | GEO `GSE90546`: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546; DOI https://doi.org/10.1016/j.cell.2016.11.048 | GEO public; explicit commercial terms not verified, `unclear` | raw/processed GEO archive | RNA pretraining, UPR biology validation |
| Dixit / Regev 2016 | scRNA-seq | mouse BMDC and human K562 | CRISPR | guide/gene perturbation, controls | time exists for BMDC stimulation; cell system varies | sample/screen metadata | GEO `GSE90063`: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063; DOI https://doi.org/10.1016/j.cell.2016.11.038 | GEO public; explicit commercial terms not verified, `unclear` | raw/processed GEO archive | RNA pretraining, held-out screen checks |
| BF-MoA | brightfield + fluorescence microscopy | cultured cells | compounds | compound, MoA, controls | dose/time/cell line from metadata tables when present | plate, well, site/FOV, z/channel, file path | Figshare DOI https://doi.org/10.17044/SCILIFELAB.21378906; metadata file `37984380`; code https://github.com/aktgpt/BF-MoA | CC BY 4.0, `commercial_ok` with attribution | metadata small; full image archive about hundreds of GB | image pretraining, MoA retrieval, bridge weak correspondence |
| RxRx1 | fluorescence microscopy | human cells | siRNA | siRNA/gene, controls | cell type verified; dose/time not primary | experiment, plate, well, site | official portal https://www.rxrx.ai/rxrx1 | Recursion non-commercial / CC BY-NC-SA-style terms, `research_only` | 125,510 PNG images plus metadata/embeddings | image pretraining, batch leakage stress test |
| JUMP Cell Painting cpg0016 | Cell Painting microscopy | human cell models | compounds, ORF, CRISPR | perturbation, controls, some target/MoA annotations | dose/time varies by source | source, batch, plate, well, site | Cell Painting Gallery https://registry.opendata.aws/cellpainting-gallery/; S3 `s3://cellpainting-gallery/cpg0016-jump/`; JUMP docs https://jump-cellpainting.github.io/ | Cell Painting Gallery data CC0 and JUMP metadata/code BSD-3-Clause, `commercial_ok` | multi-TB raw images and profiles; processed profiles available | image pretraining, image biological validation, baseline only |
| 10x Visium public examples | spatial transcriptomics + histology | human/mouse tissues | not perturbational | spot expression and image coordinates | no perturbation labels unless selected dataset has conditions | sample, spot, image scale factors | https://www.10xgenomics.com/resources/datasets | terms vary by dataset, `unclear` until selected | processed matrices, spatial coordinates, images | true paired image-expression validation only |
| SpatialLIBD | spatial transcriptomics + histology | human dorsolateral prefrontal cortex | not perturbational | spot expression, layer labels, sample IDs | no perturbation; donor/section labels | sample/section/spatial coordinates | Bioconductor https://www.bioconductor.org/packages/release/data/experiment/html/spatialLIBD.html; paper DOI https://doi.org/10.1038/s41593-020-00787-0 | package Artistic-2.0; raw-data commercial status not fully verified, `unclear` | 47,681 spots, 33,538 genes | Tier 1 paired image-expression validation |
| PERISCOPE / cpg0021 | Cell Painting + in situ sequencing | A549 / HeLa | CRISPR guides | guide/gene labels linked to images | dose/time not primary | plate, well, site, barcode calls | Cell Painting Gallery https://registry.opendata.aws/cellpainting-gallery/; S3 `s3://cellpainting-gallery/cpg0021-periscope/`; paper DOI https://doi.org/10.1038/s41592-024-02537-7 | Cell Painting Gallery CC0; article CC BY 4.0; repo permissive, `commercial_ok` | large image/profile tree | optical pooled image-genotype validation |

## Practical Use

Verified fact: no listed RNA perturbation dataset above is true same-cell
brightfield-plus-scRNA paired data. Sci-Plex3 and Perturb-seq datasets support
RNA pretraining and counterfactual validation. BF-MoA, JUMP, RxRx1, and optical
pooled screens support image-side representation learning or image-to-perturbation
validation. Spatial transcriptomics datasets provide true image-expression
pairing at spot level, but they are tissue/histology datasets, not cell-culture
brightfield perturbation screens.

Inference: the most defensible public benchmark is multi-task: RNA perturbation
counterfactuals on Sci-Plex3/Perturb-seq, morphology retrieval on BF-MoA/JUMP,
spot-level image-expression validation on SpatialLIBD or selected Visium data,
and optical pooled genotype retrieval on PERISCOPE.

Open uncertainty: commercial terms for GEO-only raw archives and specific 10x
example datasets must be checked before commercial redistribution.
