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

        self.block1 = nn.Sequential(
            nn.ConvTranspose3d(256 + 1, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose3d(128 + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose3d(64 + 1, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        e = energy.view(-1, 1)
        x = torch.cat([noise, e], dim=1)
        x = self.input_fc(x)
        x = x.view(-1, 256, 3, 3, 5)

        e3d = e.view(-1, 1, 1, 1, 1)

        e_block = e3d.expand(-1, 1, x.shape[2], x.shape[3], x.shape[4])
        x = self.block1(torch.cat([x, e_block], dim=1))

        e_block = e3d.expand(-1, 1, x.shape[2], x.shape[3], x.shape[4])
        x = self.block2(torch.cat([x, e_block], dim=1))

        e_block = e3d.expand(-1, 1, x.shape[2], x.shape[3], x.shape[4])
        x = self.block3(torch.cat([x, e_block], dim=1))

        x = x[:, :, 2:22, 2:22, :]
        return x
