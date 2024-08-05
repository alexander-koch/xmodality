import numpy as np

class SliceDS:
    def __init__(self, paths, rng, crop=True):
        self.paths = paths
        self.rng = rng
        self.crop = crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        item = np.load(self.paths[index])
        src_slice = item["src"]
        tgt_slice = item["tgt"]

        tgt_slice = (np.clip(tgt_slice, -50, 350) + 50) / 400
        min_v, max_v = src_slice.min(), src_slice.max()
        src_slice = src_slice.clip(min_v, max_v)
        src_slice = (src_slice - min_v) / (max_v - min_v)

        if self.crop:
            h = src_slice.shape[0]
            w = src_slice.shape[1]
            pad_h = 256 - h if h < 256 else 0
            pad_w = 256 - w if w < 256 else 0
            if pad_h > 0 or pad_w > 0:
                src_slice = np.pad(src_slice, ((0,pad_h), (0, pad_w)))
                tgt_slice = np.pad(tgt_slice, ((0,pad_h), (0, pad_w)))
                h = src_slice.shape[0]
                w = src_slice.shape[1]

            x = self.rng.integers(0, h-256) if h-256 > 0 else 0
            y = self.rng.integers(0, w-256) if w-256 > 0 else 0

            src_slice = src_slice[x:x+256, y:y+256]
            tgt_slice = tgt_slice[x:x+256, y:y+256]

        src_slice = np.expand_dims(src_slice, -1)
        tgt_slice = np.expand_dims(tgt_slice, -1)

        return src_slice, tgt_slice

