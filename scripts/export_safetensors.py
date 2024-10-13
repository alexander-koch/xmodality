#!/usr/bin/env python3
# Usage: ./export_safetensors.py <weights.pkl> <weights.safetensors>

import jax
import pickle
from safetensors.numpy import save_file
import numpy as np
import sys

with open(sys.argv[1], "rb") as f:
    state = pickle.load(f)
params = state.params["params"]
tensors = {}
for k,v in jax.tree_util.tree_leaves_with_path(params):
    path = "/".join([str(subkey)[2:-2] for subkey in k])
    v = v.astype(np.float32)
    tensors[path] = v

save_file(tensors, sys.argv[2])
