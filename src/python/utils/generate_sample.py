import torch
import numpy as np
from pathlib import Path

from src.python.gan.generator import Generator, NOISE_DIM

MAX_ENERGY = 500.0


def generate_sample(energy_gev:      float  = 100.0,
                    checkpoint_path: str    = "data/checkpoints/generator_final.pt",
                    output_path:     str    = "data/gan_sample.npy"):

    device = torch.device("cpu")
    G = Generator().to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()

    with torch.no_grad():
        noise  = torch.randn(1, NOISE_DIM, device=device)
        e      = torch.tensor([energy_gev / MAX_ENERGY],
                               dtype=torch.float32, device=device)
        voxels = G(noise, e).squeeze().cpu().numpy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, voxels)
    print(f"[Generate] Saved {voxels.shape} voxel grid to {output_path}")
    print(f"[Generate] Energy: {energy_gev} GeV | "
          f"Max voxel: {voxels.max():.4f} | "
          f"Total: {voxels.sum():.4f}")


if __name__ == "__main__":
    import sys
    energy = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    generate_sample(energy_gev=energy)
    