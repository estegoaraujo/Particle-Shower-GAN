import torch
import numpy as np
from pathlib import Path

from src.python.gan.generator import Generator, NOISE_DIM

MAX_ENERGY = 500.0


def generate_sample(energy_gev:      float = 100.0,
                    n_samples:        int   = 10,
                    checkpoint_path:  str   = "data/checkpoints/generator_final.pt",
                    output_dir:       str   = "data/gan_samples"):

    device = torch.device("cpu")
    G = Generator().to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    e = torch.tensor([energy_gev / MAX_ENERGY],
                      dtype=torch.float32, device=device)

    for i in range(n_samples):
        with torch.no_grad():
            noise  = torch.randn(1, NOISE_DIM, device=device)
            voxels = G(noise, e).squeeze().cpu().numpy()

        path = f"{output_dir}/gan_sample_{i:03d}.npy"
        np.save(path, voxels)
        print(f"[Generate] Sample {i:03d} — "
              f"max: {voxels.max():.4f}  total: {voxels.sum():.4f}")

    print(f"[Generate] {n_samples} samples saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    energy   = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    n        = int(sys.argv[2])   if len(sys.argv) > 2 else 10
    generate_sample(energy_gev=energy, n_samples=n)
    