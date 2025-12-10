# datasets.py
import json
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader

from model.data_utils import make_mscn_batch


class MSCNDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r") as f:
            self.examples: List[Dict[str, Any]] = json.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def build_mscn_collate_fn(ft=None, fj=None, fp=None):
    """
    Returns a collate_fn with fixed feature dims (if provided).
    If ft/fj/fp are None, make_mscn_batch will infer them from that batch.
    """
    def _collate(batch: List[Dict[str, Any]]):
        batch_samples = [ex["features"] for ex in batch]
        batch_labels  = [ex["label"]   for ex in batch]
        return make_mscn_batch(batch_samples, batch_labels, ft=ft, fj=fj, fp=fp)
    return _collate


def make_dataloaders(train_path: str = "datasets/train_features.json",
                     test_path: str = "datasets/test_features.json",
                     batch_size: int = 64,
                     num_workers: int = 0,
                     feature_dims = (0, 0, 0)):

    train_ds = MSCNDataset(train_path)
    test_ds  = MSCNDataset(test_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=build_mscn_collate_fn(feature_dims[0], feature_dims[1], feature_dims[2]),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=build_mscn_collate_fn(feature_dims[0], feature_dims[1], feature_dims[2]),
    )
    print(f"Loaded {len(train_ds)} training examples and {len(test_ds)} test examples.")

    return train_loader, test_loader
