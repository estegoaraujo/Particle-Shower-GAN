import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.python.data.loader import ShowerDataset
from src.python.gan.generator import Generator, NOISE_DIM
from src.python.gan.discriminator import Discriminator

BATCH_SIZE    = 32
EPOCHS        = 100
LR_G          = 0.0002
LR_D          = 0.0002
BETAS         = (0.5, 0.999)
CHECKPOINT_DIR = Path("data/checkpoints")
LOG_EVERY      = 10


def train(data_dir: str = "data/raw", n_showers: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ShowerDataset(data_dir, n_showers=n_showers)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, drop_last=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        loss_D_total = 0.0
        loss_G_total = 0.0

        for real_voxels, energies in loader:
            real_voxels = real_voxels.to(device)
            energies    = energies.to(device)
            batch       = real_voxels.size(0)

            real_labels = torch.ones(batch,  1, device=device)
            fake_labels = torch.zeros(batch, 1, device=device)

            noise     = torch.randn(batch, NOISE_DIM, device=device)
            fake_voxels = G(noise, energies).detach()

            D.zero_grad()
            loss_real = criterion(D(real_voxels, energies), real_labels)
            loss_fake = criterion(D(fake_voxels, energies), fake_labels)
            loss_D    = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            noise       = torch.randn(batch, NOISE_DIM, device=device)
            fake_voxels = G(noise, energies)

            G.zero_grad()
            loss_G = criterion(D(fake_voxels, energies), real_labels)
            loss_G.backward()
            opt_G.step()

            loss_D_total += loss_D.item()
            loss_G_total += loss_G.item()

        avg_D = loss_D_total / len(loader)
        avg_G = loss_G_total / len(loader)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}/{EPOCHS}]  "
                  f"Loss_D: {avg_D:.4f}  Loss_G: {avg_G:.4f}")

        if epoch % 25 == 0:
            torch.save(G.state_dict(),
                       CHECKPOINT_DIR / f"generator_epoch{epoch:03d}.pt")
            torch.save(D.state_dict(),
                       CHECKPOINT_DIR / f"discriminator_epoch{epoch:03d}.pt")
            print(f"[Epoch {epoch:03d}] Checkpoint saved.")

    torch.save(G.state_dict(), CHECKPOINT_DIR / "generator_final.pt")
    torch.save(D.state_dict(), CHECKPOINT_DIR / "discriminator_final.pt")
    print("[Train] Done. Final weights saved.")


if __name__ == "__main__":
    train()
    