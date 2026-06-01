import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.energy_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output_fc = nn.Sequential(
            nn.Linear(128 * 2 * 2 * 5 + 64, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, voxels: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(voxels)
        x = x.view(x.size(0), -1)
        e = self.energy_fc(energy.view(-1, 1))
        x = torch.cat([x, e], dim=1)
        return self.output_fc(x)
    