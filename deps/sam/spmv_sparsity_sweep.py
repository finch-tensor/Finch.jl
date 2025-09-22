import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import os
import random

num_rows = 10
num_cols = 10
density = 0.1

seed_value = 100
random.seed(seed_value)
np.random.seed(seed_value)

if not os.path.exists('spmv_sparsity_sweep'):
    os.makedirs('spmv_sparsity_sweep')
else:
    os.system("rm -rf spmv_sparsity_sweep/*")

if not os.path.exists('spmv_sparsity_sweep/MAT_FILES'):
    os.makedirs('spmv_sparsity_sweep/MAT_FILES')
else:
    os.system("rm -rf spmv_sparsity_sweep/MAT_FILES/*")
    os.makedirs('spmv_sparsity_sweep/MAT_FILES')

if not os.path.exists('spmv_sparsity_sweep/MTX_FILES'):
    os.makedirs('spmv_sparsity_sweep/MTX_FILES')
else:
    os.system("rm -rf spmv_sparsity_sweep/MTX_FILES/*")
    os.makedirs('spmv_sparsity_sweep/MTX_FILES')

matrix = sp.random(num_rows, num_cols, density, data_rvs=np.ones, random_state=seed_value)
print(matrix)

probability = 0.7  # Adjust this value to control the ratio of 1s to 0s in vector
vector = np.random.choice([0, 1], size=num_cols, p=[1 - probability, probability])
print(vector)

sio.mmwrite('spmv_sparsity_sweep/MTX_FILES/matrix.mtx', matrix)

sio.savemat('spmv_sparsity_sweep/MAT_FILES/matrix.mat', {'matrix': matrix})
sio.savemat('spmv_sparsity_sweep/MAT_FILES/vector.mat', {'vector': vector})
