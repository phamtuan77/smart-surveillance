import torch

from src.anomaly.autoencoder import ConvAutoencoder


def test_autoencoder_output_shape():
    model = ConvAutoencoder()
    model.eval()

    x = torch.rand(1, 1, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output.shape == x.shape


def test_autoencoder_output_range():
    model = ConvAutoencoder()
    model.eval()

    x = torch.rand(1, 1, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert torch.all(output >= 0)
    assert torch.all(output <= 1)