#!/usr/bin/env python3
# Usage: ./export_hdf5.py <weights.pkl> <weights.hdf5>

import pickle
import numpy as np
import h5py
import sys
import jax

with open(sys.argv[1], "rb") as f:
    state = pickle.load(f)
params = state.params["params"]

with h5py.File(sys.argv[2], "w") as f:
    for k,v in jax.tree_util.tree_leaves_with_path(params):
        path = "/".join([str(subkey)[2:-2] for subkey in k])
        v = v.astype(np.float32)
        f.create_dataset(path, data=v)

