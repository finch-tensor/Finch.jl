# Directory Structure

Here's a little roadmap to the Finch codebase! Please file an issue if this is
not up to date.

```
.
├── benchmark                  # benchmarks for internal use
│   ├── runbenchmarks.jl       # run benchmarks
│   ├── runjudge.jl            # run benchmarks on current branch and compare with main
│   └── ...
├── docs                       # documentation
│   ├── [build]                # rendered docs website
│   ├── src                    # docs website source
│   ├── fix.jl                 # fix docstrings
│   ├── examples               # example applications implemented in Finch!
│   │   └── ...
│   ├── make.jl                # build documentation locally
│   └── ...
├── ext                        # conditionally-loaded code for interaction with other packages (e.g. SparseArrays)
├── src                        # Source files
│   ├── interface              # Implementations of array api functions (e.g. map, reduce, etc.)
│   │   ├── fileio             # File IO function definitions
│   │   └── ...
│   ├── FinchLogic             # SubModule containing the High-Level IR
│   │   ├── nodes.jl           # defines the High-Level IR
│   │   └── ...
│   ├── scheduler              # Auto-Scheduler to compile High-Level IR to Finch IR
│   │   └── ...
│   ├── FinchNotation          # SubModule containing the Finch IR
│   │   ├── nodes.jl           # defines the Finch IR
│   │   ├── syntax.jl          # defines the @finch frontend syntax
│   │   └── ...
│   ├── looplets               # this is where all the Looplets live
│   ├── symbolic               # term rewriting systems for program and bounds
│   ├── tensors                # built-in Finch tensor definitions
│   │   ├── levels             # all of the levels
│   │   │   └── ...
│   │   ├── combinators        # tensor combinators which modify tensor behavior
│   │   │   └── ...
│   │   ├── tensors.jl         # defines the tensor level format interface and tensor wrapper
│   │   ├── scalars.jl         # a nice scalar type
│   │   └── masks.jl           # mask tensors (e.g. upper-triangular mask)
│   ├── transformations        # global program transformations
│   │   ├── scopes.jl          # gives unique names to indices
│   │   ├── lifetimes.jl       # adds freeze and thaw
│   │   ├── dimensionalize.jl  # computes extents for loops and declarations
│   │   ├── concordize.jl      # adds loops to ensure all accesses are concordant
│   │   └── wrapperize.jl      # converts index expressions to array wrappers
│   ├── abstract_tensor.jl     # finch array interface functions
│   ├── execute.jl             # global compiler calls
│   ├── lower.jl               # inner compiler definition
│   ├── util                   # shims and julia codegen utils (Dead code elimination, etc...)
│   │   └── ...
│   └── ...
├── test                       # tests
│   ├──  suites                # test suite files, each suite can run independently
│   ├──  reference32           # reference output for 32-bit systems
│   ├──  reference64           # reference output for 64-bit systems
│   ├──  runtests.jl           # runs the test suite. (pass -h for options and more info!)
│   └── ...
├── Project.toml               # julia-readable listing of project dependencies
├── [Manifest.toml]            # local listing of installed dependencies (don't commit this)
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```
