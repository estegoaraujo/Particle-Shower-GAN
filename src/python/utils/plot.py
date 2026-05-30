import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.python.gan.generator import Generator, NOISE_DIM
from src.python.data.loader import ShowerDataset


def load_generator(checkpoint_path: str, device: torch.device) -> Generator:
    G = Generator().to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()
    return G


def generate_fake(G: Generator, energy: float, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        noise  = torch.randn(1, NOISE_DIM, device=device)
        e      = torch.tensor([energy], dtype=torch.float32, device=device)
        voxels = G(noise, e)
    return voxels.squeeze().cpu().numpy()


def load_real(data_dir: str, idx: int = 0) -> np.ndarray:
    dataset = ShowerDataset(data_dir, normalise=True)
    voxels, _ = dataset[idx]
    return voxels.squeeze().numpy()


def plot_comparison(real: np.ndarray, fake: np.ndarray,
                    save_path: str = "data/comparison.png") -> None:
    fig = plt.figure(figsize=(14, 10), facecolor="#0A0A0A")
    fig.suptitle("Particle Shower — Real vs GAN Generated",
                 color="white", fontsize=14, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.4, wspace=0.35)

    views = [
        ("XZ projection (side)",  real.sum(axis=1),  fake.sum(axis=1)),
        ("YZ projection (side)",  real.sum(axis=0),  fake.sum(axis=0)),
        ("XY projection (front)", real.sum(axis=2),  fake.sum(axis=2)),
    ]

    vmax_real = max(v[1].max() for v in views) + 1e-8
    vmax_fake = max(v[2].max() for v in views) + 1e-8

    for col, (title, real_proj, fake_proj) in enumerate(views):
        ax_real = fig.add_subplot(gs[0, col])
        ax_fake = fig.add_subplot(gs[1, col])

        ax_real.imshow(real_proj.T, origin="lower", aspect="auto",
                       cmap="hot", vmin=0, vmax=vmax_real)
        ax_fake.imshow(fake_proj.T, origin="lower", aspect="auto",
                       cmap="hot", vmin=0, vmax=vmax_fake)

        for ax, label in [(ax_real, "Real"), (ax_fake, "GAN")]:
            ax.set_title(f"{label} — {title}", color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            ax.set_facecolor("#0A0A0A")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0A0A0A")
    plt.close()
    print(f"[Plot] Saved to {save_path}")


def plot_longitudinal(real: np.ndarray, fake: np.ndarray,
                      save_path: str = "data/longitudinal.png") -> None:
    real_profile = real.sum(axis=(0, 1))
    fake_profile = fake.sum(axis=(0, 1))

    z = np.arange(len(real_profile))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0A0A0A")
    ax.set_facecolor("#0A0A0A")

    ax.plot(z, real_profile, color="white",     lw=2, label="Real")
    ax.plot(z, fake_profile, color="#00BFFF",   lw=2, label="GAN",
            linestyle="--")

    ax.set_xlabel("Depth (voxel index, z)", color="white")
    ax.set_ylabel("Summed energy deposit", color="white")
    ax.set_title("Longitudinal Shower Profile — Real vs GAN",
                 color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1A1A1A", labelcolor="white", edgecolor="#444444")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0A0A0A")
    plt.close()
    print(f"[Plot] Saved to {save_path}")


def run(data_dir:        str = "data/raw",
        checkpoint_path: str = "data/checkpoints/generator_final.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Plot] Device: {device}")

    G    = load_generator(checkpoint_path, device)
    real = load_real(data_dir, idx=0)
    fake = generate_fake(G, energy=1.0, device=device)

    Path("data").mkdir(exist_ok=True)
    plot_comparison(real, fake)
    plot_longitudinal(real, fake)

    print("[Plot] Done.")
    print("  data/comparison.png   — 2D projections side by side")
    print("  data/longitudinal.png — longitudinal profile overlay")


if __name__ == "__main__":
    run()

