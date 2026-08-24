import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, input_size=64 * 64, latent_size=128):
        super().__init__()

        # Encoder:
        # 4096 -> 1024 -> 256 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.ReLU(),

            nn.Linear(256, latent_size)
        )

        # Decoder:
        # 128 -> 256 -> 1024 -> 4096
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),

            nn.Linear(256, 1024),
            nn.ReLU(),

            nn.Linear(1024, input_size),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        return decoded
