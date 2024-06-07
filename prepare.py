import numpy as np
import math
from pathlib import Path
import nibabel as nib
from typing import Any
from functools import reduce
from tqdm import tqdm

THRESHOLD = 200
DICE_OVERLAP = 0.3

def get_num_slices(sources):
    num_slices = []
    for source in sources:
        num_slices.append(nib.load(source).shape[-1])
    return num_slices

class MultiModalDataset:
    last_source_path: str = None
    last_target_path: str = None
    last_source: Any = None
    last_target: Any = None

    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets

        self.num_slices = get_num_slices(sources)

        bins = [[bin_index] * num for bin_index,num in enumerate(self.num_slices)]
        modulo_bins = [list(range(len(b))) for b in bins]
        self.subject_index_list = reduce(lambda x,y: x+y, bins)
        self.slice_index_list = reduce(lambda x,y: x+y, modulo_bins)

    def __len__(self):
        return sum(self.num_slices)

    def get_slice(self, src_path, tgt_path):
        if self.last_source_path is not None and self.last_source_path == src_path:
            src = self.last_source
        else:
            img = nib.load(src_path).get_fdata().astype(np.float32)

            # Src transform
            max_v = np.quantile(img, 0.999)
            #min_v = np.quantile(img, 0.01)
            min_v = 0
            img = img.clip(min_v, max_v)
            img = (img - min_v) / (max_v - min_v)

            self.last_source = img
            self.last_source_path = src_path
            src = img

        if self.last_target_path is not None and self.last_target_path == tgt_path:
            tgt = self.last_target
        else:
            img = nib.load(tgt_path).get_fdata().astype(np.float32)
            
            # Tgt transform
            img = (np.clip(img, -50, 350) + 50) / 400

            self.last_target = img
            self.last_target_path = tgt_path
            tgt = img
        return src, tgt

    def __getitem__(self, index):
        subject_index = self.subject_index_list[index]
        slice_index = self.slice_index_list[index]

        src_path = self.sources[subject_index]
        tgt_path = self.targets[subject_index]

        src_data, tgt_data = self.get_slice(src_path, tgt_path)

        src_slice = src_data[:, :, slice_index]
        tgt_slice = tgt_data[:, :, slice_index]

        return src_slice, tgt_slice

def write_ds(ds, prefix):
    print("running:", prefix)
    for i in tqdm(range(len(ds))):
        src, tgt = ds[i]
        if src.sum() == 0 or tgt.sum() == 0:
            continue
    
        # Only pick slices with sufficient similarity/overlap
        src_mask = (src > 0).astype(np.float32)
        tgt_mask = (tgt > 0).astype(np.float32)
        num_src_pixels = src_mask.sum()
        num_tgt_pixels = tgt_mask.sum()
        dice_overlap = 2 * (src_mask * tgt_mask).sum() / (num_src_pixels + num_tgt_pixels)

        if dice_overlap < DICE_OVERLAP or num_src_pixels < THRESHOLD or num_tgt_pixels < THRESHOLD:
            continue
        np.savez_compressed(f"data/{prefix}_{i}.npz", src=src, tgt=tgt)

def main():
    # Avoid data leakage
    sources = []
    targets = []
    with open("dataset.txt", "r") as f:
        for line in f:
            src, tgt = line.strip().split(",")
            sources.append(src)
            targets.append(tgt)
    indices = np.arange(len(sources))
    np.random.seed(42)
    np.random.shuffle(indices)

    index = math.floor(len(indices) * 0.7)

    train_indices = indices[:index]
    test_val_indices = indices[index:]

    val_index = math.floor(len(test_val_indices) * 0.5)
    val_indices = test_val_indices[:val_index]
    test_indices = test_val_indices[val_index:]

    print("train:", len(train_indices))
    print("val:", len(val_indices))
    print("test:", len(test_indices))

    train_sources = [sources[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]
    val_sources = [sources[i] for i in val_indices]
    val_targets = [targets[i] for i in val_indices]
    test_sources = [sources[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]

    train_ds = MultiModalDataset(train_sources, train_targets)
    val_ds = MultiModalDataset(val_sources, val_targets)
    test_ds = MultiModalDataset(test_sources, test_targets)

    Path("data").mkdir(exist_ok=True)
    write_ds(train_ds, "train")
    write_ds(val_ds, "val")
    write_ds(test_ds, "test")

if __name__ == "__main__":
    main()
