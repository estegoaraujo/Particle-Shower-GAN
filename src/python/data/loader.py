import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple



VOXEL_NX = 20   
VOXEL_NY = 20   
VOXEL_NZ = 40  


class ShowerDataset(Dataset):

    def __init__(
        self,
        data_dir:    str,
        n_showers:   Optional[int] = None,
        max_energy:  float = 500.0,
        normalise:   bool  = True,
    ):

        self.data_dir   = Path(data_dir)
        self.max_energy = max_energy
        self.normalise  = normalise

      
        csv_path = self.data_dir / "metadata.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {data_dir}. "
                                     "Run: ./build/psg --generate N first.")

        self.metadata = pd.read_csv(csv_path)

        if n_showers is not None:
            self.metadata = self.metadata.head(n_showers)

        print(f"[Dataset] Loaded metadata: {len(self.metadata)} showers "
              f"from {data_dir}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
 
        row = self.metadata.iloc[idx]
        shower_id = int(row["shower_id"])


        npy_path = self.data_dir / f"shower_{shower_id:04d}.npy"
        voxels = np.load(str(npy_path)) 

        # Sanity check
        assert voxels.shape == (VOXEL_NX, VOXEL_NY, VOXEL_NZ), \
            f"Unexpected voxel shape {voxels.shape} in {npy_path}"

 
        if self.normalise:
            total = voxels.sum()
            if total > 1e-8:
                voxels = voxels / total

        voxel_tensor = torch.from_numpy(voxels).float().unsqueeze(0)

 
        energy_tensor = torch.tensor(
            row["primary_energy_GeV"] / self.max_energy,
            dtype=torch.float32
        )

        return voxel_tensor, energy_tensor


def make_dataloader(
    data_dir:   str,
    batch_size: int  = 32,
    n_showers:  Optional[int] = None,
    shuffle:    bool = True,
    num_workers: int = 4,
) -> DataLoader:
    
    dataset = ShowerDataset(data_dir, n_showers=n_showers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), 
    )



def inspect_dataset(data_dir: str, n: int = 5) -> None:

    dataset = ShowerDataset(data_dir, n_showers=n, normalise=False)

    print(f"\n{'─'*60}")
    print(f"  Shower Dataset Inspection — {data_dir}")
    print(f"{'─'*60}")

    for i in range(min(n, len(dataset))):
        voxels, energy = dataset[i]
        raw_voxels = voxels.numpy()[0]  
        print(f"\n  Shower {i:04d}")
        print(f"    Shape:          {raw_voxels.shape}")
        print(f"    Total deposit:  {raw_voxels.sum():.4f} GeV")
        print(f"    Max voxel:      {raw_voxels.max():.6f} GeV")
        print(f"    Non-zero voxels:{(raw_voxels > 0).sum()} "
              f"/ {raw_voxels.size}")
        print(f"    Primary energy: {energy.item()*100:.1f} GeV")

    print(f"\n{'─'*60}\n")
