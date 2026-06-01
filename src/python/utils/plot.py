import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.python.gan.generator import Generator, NOISE_DIM
from src.python.data.loader import ShowerDataset

MAX_ENERGY = 500.0
ENERGIES   = [10.0, 50.0, 100.0, 200.0, 500.0]


def load_generator(checkpoint_path: str, device: torch.device) -> Generator:
    G = Generator().to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()
    return G


def generate_fake(G: Generator, energy_gev: float,
                  device: torch.device) -> np.ndarray:
    with torch.no_grad():
        noise = torch.randn(1, NOISE_DIM, device=device)
        e     = torch.tensor([energy_gev / MAX_ENERGY],
                              dtype=torch.float32, device=device)
        voxels = G(noise, e)
    return voxels.squeeze().cpu().numpy()


def load_real_by_energy(data_dir: str, energy_gev: float) -> np.ndarray:
    dataset = ShowerDataset(data_dir, normalise=True, max_energy=MAX_ENERGY)
    for i in range(len(dataset)):
        voxels, e = dataset[i]
        if abs(e.item() * MAX_ENERGY - energy_gev) < 1.0:
            return voxels.squeeze().numpy()
    raise ValueError(f"No shower found at {energy_gev} GeV in {data_dir}")


def plot_comparison(real: np.ndarray, fake: np.ndarray,
                    energy_gev: float,
                    save_path: str = None) -> None:
    if save_path is None:
        save_path = f"data/comparison_{int(energy_gev)}GeV.png"

    fig = plt.figure(figsize=(14, 10), facecolor="#0A0A0A")
    fig.suptitle(
        f"Particle Shower — Real vs GAN Generated  ({energy_gev} GeV)",
        color="white", fontsize=14, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    views = [
        ("XZ projection (side)",  real.sum(axis=1), fake.sum(axis=1)),
        ("YZ projection (side)",  real.sum(axis=0), fake.sum(axis=0)),
        ("XY projection (front)", real.sum(axis=2), fake.sum(axis=2)),
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
    print(f"[Plot] Saved {save_path}")


def plot_longitudinal_multi(G: Generator, data_dir: str,
                             device: torch.device,
                             save_path: str = "data/longitudinal_multi.png") -> None:
    colors_real = ["#555555", "#777777", "#999999", "#BBBBBB", "#FFFFFF"]
    colors_fake = ["#003080", "#0055CC", "#0088FF", "#44AAFF", "#88CCFF"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0A0A0A")
    ax.set_facecolor("#0A0A0A")

    z = np.arange(40)

    for i, energy in enumerate(ENERGIES):
        real  = load_real_by_energy(data_dir, energy)
        fake  = generate_fake(G, energy, device)

        real_profile = real.sum(axis=(0, 1))
        fake_profile = fake.sum(axis=(0, 1))

        ax.plot(z, real_profile, color=colors_real[i], lw=2,
                label=f"Real {int(energy)} GeV")
        ax.plot(z, fake_profile, color=colors_fake[i], lw=2,
                linestyle="--", label=f"GAN {int(energy)} GeV")

    ax.set_xlabel("Depth (voxel index, z)", color="white")
    ax.set_ylabel("Summed energy deposit (normalised)", color="white")
    ax.set_title("Longitudinal Profiles — All Energies", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1A1A1A", labelcolor="white",
               edgecolor="#444444", fontsize=7, ncol=2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0A0A0A")
    plt.close()
    print(f"[Plot] Saved {save_path}")


def run(data_dir:        str = "data/raw",
        checkpoint_path: str = "data/checkpoints/generator_final.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Plot] Device: {device}")

    G = load_generator(checkpoint_path, device)

    Path("data").mkdir(exist_ok=True)

    for energy in ENERGIES:
        try:
            real = load_real_by_energy(data_dir, energy)
            fake = generate_fake(G, energy, device)
            plot_comparison(real, fake, energy)
        except ValueError as e:
            print(f"[Plot] Skipping {energy} GeV: {e}")

    plot_longitudinal_multi(G, data_dir, device)
    print("[Plot] Done.")


if __name__ == "__main__":
    run()
    