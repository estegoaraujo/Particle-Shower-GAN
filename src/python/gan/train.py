import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.python.data.loader import ShowerDataset
from src.python.gan.generator import Generator, NOISE_DIM
from src.python.gan.discriminator import Discriminator

BATCH_SIZE     = 64
EPOCHS         = 100
LR_G           = 0.0001
LR_D           = 0.0001
BETAS          = (0.0, 0.9)
LAMBDA_GP      = 10.0
N_CRITIC       = 2
CHECKPOINT_DIR = Path("data/checkpoints")
LOG_EVERY      = 5


def gradient_penalty(D: nn.Module,
                     real: torch.Tensor,
                     fake: torch.Tensor,
                     energy: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    critic_interp = D(interpolated, energy)

    gradients = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1.0) ** 2).mean()
    return penalty


def train(data_dir: str = "data/raw", n_showers: int = None):
    device = torch.device("cpu")

    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)

    print(f"[Train] Device: {device}")
    print(f"[Train] Threads: {torch.get_num_threads()}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ShowerDataset(data_dir, n_showers=n_showers, max_energy=500.0)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=6,
                          pin_memory=False, drop_last=True,
                          persistent_workers=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    torch.jit.optimize_for_inference

    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=EPOCHS)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        loss_D_total = 0.0
        loss_G_total = 0.0
        gp_total     = 0.0

        for real_voxels, energies in loader:
            real_voxels = real_voxels.to(device)
            energies    = energies.to(device)
            batch       = real_voxels.size(0)

            for _ in range(N_CRITIC):
                noise       = torch.randn(batch, NOISE_DIM, device=device)
                fake_voxels = G(noise, energies).detach()

                gp     = gradient_penalty(D, real_voxels, fake_voxels,
                                           energies, device)
                loss_D = (D(fake_voxels, energies).mean()
                          - D(real_voxels, energies).mean()
                          + LAMBDA_GP * gp)

                D.zero_grad()
                loss_D.backward()
                opt_D.step()

            noise       = torch.randn(batch, NOISE_DIM, device=device)
            fake_voxels = G(noise, energies)
            loss_G      = -D(fake_voxels, energies).mean()

            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loss_D_total += loss_D.item()
            loss_G_total += loss_G.item()
            gp_total     += gp.item()

        sched_G.step()
        sched_D.step()

        avg_D  = loss_D_total / len(loader)
        avg_G  = loss_G_total / len(loader)
        avg_gp = gp_total     / len(loader)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}/{EPOCHS}]  "
                  f"W-Loss: {-avg_D:.4f}  "
                  f"Loss_G: {avg_G:.4f}  "
                  f"GP: {avg_gp:.4f}")

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
