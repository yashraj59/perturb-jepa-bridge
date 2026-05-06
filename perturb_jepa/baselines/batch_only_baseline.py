from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from perturb_jepa.baselines.metadata_only_retrieval import metadata_only_retrieval_metrics

TECHNICAL_METADATA_COLUMNS = (
    "batch",
    "plate",
    "run",
    "well",
    "site",
    "z_plane",
    "imaging_channel",
    "sequencing_lane",
    "library_id",
)


def batch_only_retrieval_metrics(
    query_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    gallery_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    label_col: str = "condition_key",
    columns: Sequence[str] = TECHNICAL_METADATA_COLUMNS,
    ks: Sequence[int] = (1, 5, 10),
    prefix: str = "batch_only",
) -> dict[str, float]:
    """Retrieval baseline using only technical metadata.

    This is a leakage diagnostic: strong biological retrieval here usually means
    condition labels are confounded with acquisition/batch metadata.
    """

    return metadata_only_retrieval_metrics(
        query_metadata,
        gallery_metadata,
        label_col=label_col,
        columns=columns,
        ks=ks,
        prefix=prefix,
    )
