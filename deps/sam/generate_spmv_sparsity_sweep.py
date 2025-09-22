# script to generate 50 random 3D tensors (seeded, produces same 50 each time)
import numpy as np
import random
import os
import scipy.io as sio
import scipy.sparse as sps

# from scipy.io import mmread

# Set the seed value
# previously used to be this: seed_value = 42
seed_value = 100
random.seed(seed_value)
np.random.seed(seed_value)

# generating matrix dimensions and storing results in an array, array size is 2, 1 matrix and 2 dimensions per matrix

# conditions which need to be met for each set of 3 tensor dimensions: no dimension can't be 0,
# and can't have a tensor with more than 900 elements (meaning dimension1*dimension2*dimension3 <= 900)
# note try to make it so no dimension is 1 or 2 (gives slight issues later, esp 2nd and 3rd dimensions)
dimensions = [0] * 2
dimensions_onematrix = [0] * 2

# x goes from 0 to __ (before 0 to 49)
for x in range(1):
    # dimensions_onematrix[0] = random.randint(1,60)
    # dimensions_onematrix[1] = random.randint(3,60)

    # while((dimensions_onetensor[0]*dimensions_onetensor[1]*dimensions_onetensor[2])>901):
    #     dimensions_onematrix[0] = random.randint(1,60)
    #     dimensions_onematrix[1] = random.randint(3,60)
    #     dimensions_onematrix[2] = random.randint(3,60)
    dimensions_onematrix[0] = 10
    dimensions_onematrix[1] = 10

    dimensions[x * 3] = dimensions_onematrix[0]
    dimensions[(x * 3) + 1] = dimensions_onematrix[1]

    dimensions_onematrix[0] = 0
    dimensions_onematrix[1] = 0
    # print('\n')


# Generating matrix values based on the dimensions now stored in the dimensions (2 elem) array
# i goes from 0 to __ (before 0 to 49)
matrix_num = 1
randomNumber = 0
numToInsert = 0
countnnz = 0
# can add in as many sparsity numbers here (num elements in the sparsities array = num matrices being generated)
sparsities = [0.5]
# NEED TO CHANGE suitesparse_path for this to work: frostt_path = os.environ['FROSTT_PATH']
ss_path = ""
for i in range(1):
    filename = os.path.join(ss_path, "rand_matrix" + str(matrix_num) + ".mtx")
    sparsity = sparsities[i]
    f = open(filename, "w")
    f.write("\n")
    lineToAddInFile = ""
    arr = np.zeros([dimensions[i * 3], dimensions[(i * 3) + 1]], dtype=int)
    for x in range(len(arr)):
        for y in range(len(arr[x])):
            # TO CHANGE SPARSITY: generate random number from 1 to 9; if 1,2,3,7,8,9 don't add a num in, only add if 4,5,6
            # randomNumber = random.randint(1,9)
            randomNumber = random.random()
            if randomNumber > sparsity:
                numToInsert = random.randint(1, 100)
                arr[x][y] = numToInsert
                numToInsert = 0
                randomNumber = 0
            # print(arr[x][y][z])
            if arr[x][y] != 0:
                # tensor files are not 0 indexed - say want to insert a point at (0,0,0),
                # then need to feed in (1,1,1) to the tensor file to insert at the (0,0,0)
                # location
                lineToAddInFile = (
                    "" + str(x + 1) + " " + str(y + 1) + " " + str(arr[x][y])
                )
                countnnz += 1
                f.write(lineToAddInFile + "\n")
    # writing in first line in file:
    with open(filename, "r") as f:
        content = f.read()
    updated_content = (
        "" +
        str(dimensions[i * 3]) +
        " " +
        str(dimensions[i * 3 + 1]) +
        " " +
        str(countnnz) +
        content
    )
    with open(filename, "w") as f:
        f.write(updated_content)

    with open(filename, "r") as file:
        data = file.readlines()

    header = data.pop(0)
    num_rows, num_cols, num_nonzeros = map(int, header.strip().split())
    matrix_data = []
    row_indices = []
    col_indices = []
    for line in data:
        row, col, value = map(float, line.strip().split())
        row_indices.append(int(row) - 1)  # Convert to 0-based indexing
        col_indices.append(int(col) - 1)  # Convert to 0-based indexing
        matrix_data.append(value)
    matrix = sps.coo_matrix(
        (matrix_data, (row_indices, col_indices)), shape=(num_rows, num_cols)
    )
    output_file = os.path.join(ss_path, "rand_matrix" + str(matrix_num) + ".mat")
    sio.savemat(output_file, {"matrix": matrix}, do_compression=True)

    # vec = sps.random(dimensions[i*3+1], 1, 0, data_rvs=np.ones)
    vec = np.ones(dimensions[i * 3 + 1])
    output_file1 = os.path.join(ss_path, "rand_vector" + str(matrix_num) + ".mat")
    sio.savemat(output_file1, {"vector": vec}, do_compression=True)

    # f.close()
    # a = mmread(filename)
    # a.toarray()
    # scipy.io.savemat("rand_matrix"+str(matrix_num)+".mat", {'mydata': a})

    # f.write(""+str(dimensions[i*3]) + " " + str(dimensions[i*3+1]) + " " + str(countnnz))
    # f.write("\n")
    matrix_num = matrix_num + 1


# first step: one randomly generated 3D tensor given first set dimensions
# Note: generally if 2/3 elems in a tensor is 0, it can be considered sparse
# approach: 2/3 of the time add in a 0, 1/3 of the time add in an integer from 0 to 100
# (use randint to generate num from 1 to 9 inclusive, and depending on where the num is, insert number or not)
# print('dimensions:')
# print(dimensions[0])
# print(dimensions[1])
# print(dimensions[2])
# print('tensor vals')

"""
arr = np.zeros([dimensions[0],dimensions[1],dimensions[2]], dtype=int)
randomNumber = 0
numToInsert = 0
for x in range(len(arr)):
    for y in range(len(arr[x])):
        for z in range(len(arr[x][y])):
            #generate random number from 1 to 9; if 1,2,3,7,8,9 don't add a num in, only add if 4,5,6
            randomNumber = random.randint(1,9)
            if(randomNumber==4 or randomNumber==5 or randomNumber==6):
                numToInsert = random.randint(1,100)
                arr[x][y][z] = numToInsert
                numToInsert = 0
            print(arr[x][y][z])

            #lineToAddInFile="" + str(x) + " " + str(y) + " " + str(z) + " " + str(arr[x][y][z])
            #f.write(lineToAddInFile + '\n')

print('dimensions:')
print(dimensions[3])
print(dimensions[4])
print(dimensions[5])
print('tensor vals')
arr = np.zeros([dimensions[3],dimensions[4],dimensions[5]], dtype=int)
randomNumber = 0
numToInsert = 0
for x in range(len(arr)):
    for y in range(len(arr[x])):
        for z in range(len(arr[x][y])):
            #generate random number from 1 to 9; if 1,2,3,7,8,9 don't add a num in, only add if 4,5,6
            randomNumber = random.randint(1,9)
            if(randomNumber==4 or randomNumber==5 or randomNumber==6):
                numToInsert = random.randint(1,100)
                arr[x][y][z] = numToInsert
                numToInsert = 0
                randomNumber = 0
            print(arr[x][y][z])

            #lineToAddInFile="" + str(x) + " " + str(y) + " " + str(z) + " " + str(arr[x][y][z])
            #f.write(lineToAddInFile + '\n')
"""
