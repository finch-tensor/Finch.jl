# The Sparse Abstract Machine (SAM) IR, Compiler, and Simulator 

![Master Makefile CI](https://github.com/weiya711/sam/actions/workflows/makefile.yml/badge.svg?branch=master)
![Master Python CI](https://github.com/weiya711/sam/actions/workflows/python-package-conda.yml/badge.svg?branch=master)

## SAM Front-end Compiler

Overview:
tensor expression + format language + schedule
-->
SAM Graph 
--> 
dot file and png of dot file
-->        
RTL Graph or Simulator Graph

### Compiling SAM graphs
Init the taco/ repo as a submodule
```
make submodules
```

Setup the compilation for the taco/ repo
```
make taco/build
```

Run the script to generate a handful of example sam graphs
```
make sam
```

The example sam graphs should now be located in `compiler/sam-outputs/` in both the `dot/` and `png/` folers. 

### Naming convention
Naming rules
- all (block) types are lower case: repeat, repeat_gen, fiber_lookup, fiber_write, reduce, intersect, union, sparse_accum
- network signal types are: crd, ref, val, repsig, and bv
- Tensor casing: Matrices and higher order tensors are upper case, scalars and vectors are lower case
- Index variables are going to be i, j, k, ..., etc.
- Tensor ranks are going to correspond to 0, 1, 2, ..., etc. (no longer using rows and columns)
- For a given expression result is always 'x' (or 'X') and the inputs start from 'b, c, ..., etc.' of equivalently 'B, C, ..., etc.'

Metadata Naming
Metadata naming convention for other blocks: <block name>-<metadata>
Metadata naming convention for fiber (lookup and write) blocks: fiber_<lookup|write>-<tensor>_<index>_<format>_<glb?>
Examples:
1. fiber_lookup_Bi_B0_compressed
2. repeat_Ci
 
## SAM Simulator
### Installing SAM Simulator as a Package
 ```
 pip install -e .
 ```
 
### Running Tests
The simulator uses pytest to run tests

To run all tests type 
```
cd sam/sim/
pytest
```

Use the following pytest optional arguments below
```
--debug-sim                 Turn on debug mode for sim
--count=<n>                 Repeat each test for n iterations 
-k <testname>[<paramlist>]  Run only tests with testname and paramlist
-vv                         Double verbose
-s                          Forward printouts to stdout
--full-trace                Print full trace to stdout
```


### Test Naming Convention
Full kernel tests follow the naming convention `test_<kernel>_<rand|direct>_<outformat>_<in1format>_<in2format>_...<innformat>` where: 
1. `<kernel>` is the name of the tensor algebra kernel being tested (e.g. mat_elemmul, mat_mul, vec_elemmul, etc.)
2. `*format` takes on `u | c | s` for formats uncompressed, compressed, singleton respectively
3. `<rand|direct> specifies if the test is _randomly generated_ or a _directed (handwritten)_ test

Primitive unit tests follow the naming convention `test_<primitive>_<feature>_<order>` where:
1. `<primitive>` is the name of the primitive being tested (e.g. array, intersect, union, etc.)
2. `<feature>` is the name of the feature being tested (e.g. for an array we can test both loads and stores)
3. `<order>` is the name of the order of stream being tested (1d for vectors,
2d for matrices, ..., and nd for all dimensions/tensor orders, etc.)
 

### Directory Structure
```
sim
│   
│       
│
└───src
│   │   base.py
│   │   joiner.py
│   │   ...         # All primitive block classes
│   
└───test
│   │   test.py
│   │   file022.txt
│   │
│   └───apps
│       │   test_mat_elemmul.py
│       │   ...     # Full kernel/expression tests
│
└───────primitives
        │   test_joiner.py
        │   ...     # Primitive unit tests
   
```

## SAM Binding to Onyx
See the `README` in `sam/sam/onyx`

## License
All files in this project (code, scripts, documentaiton) are released under the [MIT License](LICENSE) 
