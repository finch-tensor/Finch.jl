# SAM Binding to the Onyx CGRA
Part of the [Stanford AHA project](https://aha.stanford.edu/)
The Stanford AHA Github can be found [here](https://github.com/StanfordAHA/)

## Running SuiteSparse Matrices in the Onyx CGRA
From the [garnet repo](https://github.com/StanfordAHA/garnet) (in the aha docker) in the `spVspV\_file\_name` branch run the following command:
```
python ../sam/sam/onyx/suitesparse.py --docker --matrix_file <txtname> --benchname <benchname> 
```
where `<txtname>` is a text file with each SuiteSparse matrix name that should be
run on a newline. Examples of `<txtname>` files can be found in
`sam/scripts/tensor_names/*.txt` 

where `<benchname>` is the name of the benchmark. Currently the following benchnames are supported: `["matmul_ijk", "mat_elemmul", "mat_elemadd", "mat_elemadd3"]`
