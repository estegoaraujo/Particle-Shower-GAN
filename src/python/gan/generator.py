import torch
import torch.nn as nn

NOISE_DIM = 128
VOXEL_SHAPE = (1, 20, 20, 40)

class Generator(nn.Module):
    def __init__(self, noise_dim: int = NOISE_DIM):
        super().__init__()
        self.noise_dim = noise_dim

        self.input_fc = nn.Sequential(
            nn.Linear(noise_dim + 1, 256 * 3 * 3 * 5),
            nn.BatchNorm1d(256 * 3 * 3 * 5),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        energy = energy.view(-1, 1)
        x = torch.cat([noise, energy], dim=1)
        x = self.input_fc(x)
        x = x.view(-1, 256, 3, 3, 5)
        x = self.conv_blocks(x)
        x = x[:, :, 2:22, 2:22, :]
        return x
    