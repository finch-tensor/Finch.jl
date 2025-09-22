import numpy as np
import os
import glob
import shutil
from scripts.util.util import FormatWriter, InputCacheSuiteSparse

# PARAMS
tile = True
app_name = "tensor3_ttv"
vector_names = ['c']


tiled_tensors = glob.glob(f"tiles/{app_name}/mtx/*.tns")
formatwriter = FormatWriter()
inputCache = InputCacheSuiteSparse()

for tensor in tiled_tensors:
    if any(x in tensor for x in vector_names):
        # vector
        inputCache.load(tensor)
        formatwriter.writeout_separate_sparse_only()
    else:
        print("regular 3d tensors can be packed and tiled")
