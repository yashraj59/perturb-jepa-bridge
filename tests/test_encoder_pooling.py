import torch
import pytest

from perturb_jepa.models.image_encoder import ImageEncoder, ImageEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoder, RNAEncoderConfig


def test_rna_encoder_mean_token_pooling_changes_cell_embedding():
    torch.manual_seed(0)
    gene_ids = torch.arange(6).expand(2, -1)
    expression = torch.randn(2, 6)
    cls_encoder = RNAEncoder(RNAEncoderConfig(vocab_size=6, dim=8, depth=1, heads=2, max_genes=6, pooling="cls"))
    mean_encoder = RNAEncoder(RNAEncoderConfig(vocab_size=6, dim=8, depth=1, heads=2, max_genes=6, pooling="mean_tokens"))
    mean_encoder.load_state_dict(cls_encoder.state_dict())
    cls_encoder.eval()
    mean_encoder.eval()

    cls_output = cls_encoder(gene_ids, expression)
    mean_output = mean_encoder(gene_ids, expression)

    assert cls_output.cell_embedding.shape == (2, 8)
    assert mean_output.cell_embedding.shape == (2, 8)
    assert not torch.allclose(cls_output.cell_embedding, mean_output.cell_embedding)
    torch.testing.assert_close(mean_output.cell_embedding, mean_output.token_embeddings.mean(dim=1))


def test_image_encoder_mean_patch_pooling_changes_image_embedding():
    torch.manual_seed(0)
    images = torch.randn(2, 1, 8, 8)
    cls_encoder = ImageEncoder(ImageEncoderConfig(in_channels=1, image_size=8, patch_size=4, dim=8, depth=1, heads=2, pooling="cls"))
    mean_encoder = ImageEncoder(
        ImageEncoderConfig(in_channels=1, image_size=8, patch_size=4, dim=8, depth=1, heads=2, pooling="mean_patches")
    )
    mean_encoder.load_state_dict(cls_encoder.state_dict())
    cls_encoder.eval()
    mean_encoder.eval()

    cls_output = cls_encoder(images)
    mean_output = mean_encoder(images)

    assert cls_output.image_embedding.shape == (2, 8)
    assert mean_output.image_embedding.shape == (2, 8)
    assert not torch.allclose(cls_output.image_embedding, mean_output.image_embedding)
    torch.testing.assert_close(mean_output.image_embedding, mean_output.patch_embeddings.mean(dim=1))


def test_encoder_pooling_rejects_unknown_modes():
    rna = RNAEncoder(RNAEncoderConfig(vocab_size=4, dim=8, depth=1, heads=2, max_genes=4, pooling="bad"))
    with pytest.raises(ValueError, match="unsupported RNA pooling"):
        rna(torch.arange(4).expand(1, -1), torch.ones(1, 4))

    image = ImageEncoder(ImageEncoderConfig(in_channels=1, image_size=8, patch_size=4, dim=8, depth=1, heads=2, pooling="bad"))
    with pytest.raises(ValueError, match="unsupported image pooling"):
        image(torch.ones(1, 1, 8, 8))
