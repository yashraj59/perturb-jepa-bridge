from __future__ import annotations

from dataclasses import dataclass
import tarfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from perturb_jepa.data.conditions import MetadataVocab, build_condition_bags
from perturb_jepa.data.schema import IMAGE_COLUMNS, normalize_image_manifest, normalize_value


BRIGHTFIELD_MARKERS = ("bf", "brightfield", "phase", "ch06", "ch07", "ch08")


def read_image_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize_image_manifest(frame)


def filter_label_free_channels(
    manifest: pd.DataFrame,
    *,
    channel_col: str = "channel_or_z",
    markers: Iterable[str] = BRIGHTFIELD_MARKERS,
) -> pd.DataFrame:
    if channel_col not in manifest.columns:
        raise ValueError(f"{channel_col!r} not found in image manifest")
    lowered = manifest[channel_col].astype(str).str.lower()
    mask = lowered.apply(lambda value: any(marker in value for marker in markers))
    return manifest.loc[mask].copy()


def _find_column(frame: pd.DataFrame, candidates: Iterable[str], *, default: str | None = None) -> str | None:
    lower_to_original = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return default


def bf_moa_table_to_manifest(
    bf_data: pd.DataFrame,
    *,
    image_root: str | Path = "",
) -> pd.DataFrame:
    """Convert BF-MoA bf_data.csv into the normalized long image manifest."""

    plate_col = _find_column(bf_data, ("plate", "Metadata_Plate", "Metadata_PlateName"))
    well_col = _find_column(bf_data, ("well", "Metadata_Well"))
    site_col = _find_column(bf_data, ("site", "Metadata_Site", "field"))
    compound_col = _find_column(bf_data, ("compound", "Compound", "Metadata_Compound"))
    moa_col = _find_column(bf_data, ("moa", "MoA", "Metadata_MoA"))
    target_col = _find_column(bf_data, ("target_gene", "target", "Metadata_Target"), default=None)
    channel_cols = [column for column in bf_data.columns if column.upper() in {f"C{i}" for i in range(1, 7)}]
    if not channel_cols:
        raise ValueError("No BF-MoA channel columns C1-C6 were found")

    rows: list[dict[str, str]] = []
    root = Path(image_root)
    for _, source in bf_data.iterrows():
        plate = normalize_value(source.get(plate_col, "unknown") if plate_col else "unknown")
        well = normalize_value(source.get(well_col, "unknown") if well_col else "unknown")
        site = normalize_value(source.get(site_col, "unknown") if site_col else "unknown")
        compound = normalize_value(source.get(compound_col, "unknown") if compound_col else "unknown")
        moa = normalize_value(source.get(moa_col, "unknown") if moa_col else "unknown")
        target = normalize_value(source.get(target_col, "") if target_col else "")
        for channel_col in channel_cols:
            image_name = normalize_value(source[channel_col])
            image_path = str(root / plate / image_name) if str(root) else image_name
            rows.append(
                {
                    "image_path": image_path,
                    "plate": plate,
                    "well": well,
                    "site": site,
                    "channel_or_z": channel_col,
                    "perturbation": compound,
                    "compound": compound,
                    "moa": moa,
                    "target_gene": target,
                    "perturbation_type": "compound",
                    "dose": "10uM",
                    "time": "48h",
                    "cell_line": "U2OS",
                    "batch": plate,
                }
            )
    return normalize_image_manifest(pd.DataFrame(rows, columns=[*IMAGE_COLUMNS, "perturbation_type"]))


def extract_bf_moa_metadata(data_tables_tar: str | Path, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(data_tables_tar, "r:gz") as archive:
        members = [member for member in archive.getmembers() if Path(member.name).name == "bf_data.csv"]
        if not members:
            raise ValueError("bf_data.csv not found in data_tables archive")
        archive.extract(members[0], path=output)
        return output / members[0].name


def image_array_to_chw_float(array: np.ndarray, *, channels: int | None = None) -> np.ndarray:
    original_dtype = array.dtype
    image = np.asarray(array)
    if image.ndim == 2:
        image = image[None, :, :]
    elif image.ndim == 3:
        expected_channels = {channels} if channels is not None else {1, 2, 3, 4}
        first_is_channel = image.shape[0] in expected_channels
        last_is_channel = image.shape[-1] in expected_channels
        if last_is_channel and not first_is_channel:
            image = np.moveaxis(image, -1, 0)
        elif first_is_channel and not last_is_channel:
            image = image
        elif last_is_channel and image.shape[-1] in {1, 3}:
            image = np.moveaxis(image, -1, 0)
        elif first_is_channel:
            image = image
        else:
            raise ValueError("3D image arrays must be channel-first or channel-last")
    else:
        raise ValueError("image arrays must have shape [H, W], [H, W, C], or [C, H, W]")

    if channels is not None and image.shape[0] != channels:
        if image.shape[0] == 1 and channels > 1:
            image = np.repeat(image, channels, axis=0)
        elif image.shape[0] > channels:
            image = image[:channels]
        else:
            raise ValueError(f"image has {image.shape[0]} channels, expected {channels}")

    image = image.astype(np.float32, copy=False)
    if np.issubdtype(original_dtype, np.integer):
        image = image / float(np.iinfo(original_dtype).max)
    return image


def load_image_array(
    path: str | Path,
    *,
    channels: int | None = None,
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        if resize is not None:
            raise ValueError("resize is only supported for Pillow-backed image files")
        return image_array_to_chw_float(np.load(path), channels=channels)

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install the data extra to read image files: pip install -e '.[data]'") from exc

    with Image.open(path) as image:
        if channels == 1:
            image = image.convert("L")
        elif channels == 3:
            image = image.convert("RGB")
        if resize is not None:
            image = image.resize(resize)
        return image_array_to_chw_float(np.asarray(image), channels=channels)


@dataclass(frozen=True)
class ImageManifestExample:
    image: np.ndarray
    image_path: str
    condition_key: str
    perturbation_id: int
    perturbation_type_id: int
    cell_line_id: int
    batch_id: int
    dose: float
    time: float


@dataclass
class ImageManifestBatch:
    images: torch.Tensor
    image_patch_mask: torch.Tensor
    perturbation_id: torch.Tensor
    perturbation_type_id: torch.Tensor
    cell_line_id: torch.Tensor
    batch_id: torch.Tensor
    dose: torch.Tensor
    time: torch.Tensor
    condition_key: list[str]
    image_path: list[str]


class ImageManifestDataset(Dataset):
    """Load normalized image manifest rows as channel-first float tensors."""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        *,
        image_root: str | Path = "",
        channels: int | None = None,
        resize: tuple[int, int] | None = None,
        transform: Callable[[np.ndarray], np.ndarray | torch.Tensor] | None = None,
        metadata_vocab: MetadataVocab | None = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            self.manifest = read_image_manifest(manifest)
        else:
            self.manifest = normalize_image_manifest(manifest)
        self.image_root = Path(image_root)
        self.channels = channels
        self.resize = resize
        self.transform = transform
        self.metadata_vocab = metadata_vocab or MetadataVocab.from_frame(self.manifest)
        self.encoded_manifest = self.metadata_vocab.encode_frame(self.manifest)
        self.condition_bags = build_condition_bags(self.manifest)

    def __len__(self) -> int:
        return len(self.manifest)

    def _resolve_path(self, image_path: str | Path) -> Path:
        path = Path(image_path)
        if path.is_absolute() or str(self.image_root) == ".":
            return path
        if str(self.image_root):
            return self.image_root / path
        return path

    def __getitem__(self, index: int) -> ImageManifestExample:
        row = self.encoded_manifest.iloc[index]
        resolved = self._resolve_path(str(row["image_path"]))
        image = load_image_array(resolved, channels=self.channels, resize=self.resize)
        if self.transform is not None:
            image = self.transform(image)
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
        return ImageManifestExample(
            image=np.asarray(image, dtype=np.float32),
            image_path=str(row["image_path"]),
            condition_key=str(row["condition_key"]),
            perturbation_id=int(row["perturbation_id"]),
            perturbation_type_id=int(row["perturbation_type_id"]),
            cell_line_id=int(row["cell_line_id"]),
            batch_id=int(row["batch_id"]),
            dose=float(row["dose"]),
            time=float(row["time"]),
        )


class ImageManifestCollator:
    def __init__(
        self,
        *,
        patch_size: int | None = 8,
        patch_mask_prob: float = 0.15,
        seed: int | None = None,
    ) -> None:
        if patch_size is not None and patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not 0.0 <= patch_mask_prob <= 1.0:
            raise ValueError("patch_mask_prob must be between 0 and 1")
        self.patch_size = patch_size
        self.patch_mask_prob = patch_mask_prob
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __call__(self, examples: Sequence[ImageManifestExample]) -> ImageManifestBatch:
        if not examples:
            raise ValueError("cannot collate an empty image batch")
        images = [torch.as_tensor(example.image, dtype=torch.float32) for example in examples]
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError(f"all images in a batch must have the same shape, got {sorted(shapes)}")
        image_batch = torch.stack(images, dim=0)
        patch_mask = self._make_patch_mask(image_batch)
        return ImageManifestBatch(
            images=image_batch,
            image_patch_mask=patch_mask,
            perturbation_id=torch.tensor([example.perturbation_id for example in examples], dtype=torch.long),
            perturbation_type_id=torch.tensor(
                [example.perturbation_type_id for example in examples],
                dtype=torch.long,
            ),
            cell_line_id=torch.tensor([example.cell_line_id for example in examples], dtype=torch.long),
            batch_id=torch.tensor([example.batch_id for example in examples], dtype=torch.long),
            dose=torch.tensor([example.dose for example in examples], dtype=torch.float32),
            time=torch.tensor([example.time for example in examples], dtype=torch.float32),
            condition_key=[example.condition_key for example in examples],
            image_path=[example.image_path for example in examples],
        )

    def _make_patch_mask(self, images: torch.Tensor) -> torch.Tensor:
        if self.patch_size is None:
            return torch.zeros((images.shape[0], 0), dtype=torch.bool)
        _, _, height, width = images.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("image height and width must be divisible by patch_size")
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        if self.patch_mask_prob == 0.0:
            return torch.zeros((images.shape[0], num_patches), dtype=torch.bool)
        return torch.rand((images.shape[0], num_patches), generator=self.generator) < self.patch_mask_prob
