We welcome contributions to Finch, and follow the [Julia contributing
guidelines](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md).  If
you use or want to use Finch and have a question or bug, please do file a
[Github issue](https://github.com/finch-tensor/Finch.jl/issues)!  If you want
to contribute to Finch, please first file an issue to double check that there is
interest from a contributor in the feature.

## Versions

The Finch API is stable and
we follow [semantic
versioning](https://semver.org/) conventions.
The main branch of the Finch repo is the most up-to-date development branch.
While it is not stable, it should always pass tests.

Contributors will develop and test Finch from a local directory. Please see the
[Package documentation](https://pkgdocs.julialang.org/v1/getting-started/) for more
info, particularly the section on [developing](https://pkgdocs.julialang.org/v1/managing-packages/#developing).

To determine which version of
Finch you have, run `Pkg.status("Finch")` in the Julia REPL. If the installed
version of Finch tracks a local path, the output will include the path like so:

```
Status `~/.julia/environments/v1.9/Project.toml`
  [9177782c] Finch v0.5.4 `~/Projects/Finch.jl`
```

If the installed version of Finch tracks a particular version (probably not what
you want since it will not reflect local changes), the output will look like this:

```
Status `~/.julia/environments/v1.8/Project.toml`
  [9177782c] Finch v0.5.4
```

## Utilities

Finch include several scripts that can be executed directly, e.g. `runtests.jl`.
These scripts are all have local [Pkg
environments](https://pkgdocs.julialang.org/v1/getting-started/#Getting-Started-with-Environments).
The scripts include convenience headers to automatically use their respective
environments, so you won't need to worry about `--project=.` flags, etc.

## Testing

All pull requests should pass continuous integration testing before merging.
The test suite has a few options, which are accessible through running the test
suite directly as `./test/runtests.jl`.

Finch compares compiler output against reference versions.

If you have the appropriate permissions, you can run the
[FixBot](https://github.com/finch-tensor/Finch.jl/actions/workflows/FixBot.yml)
github action on your PR branch to automatically generate output for both 32-bit
and 64-bit builds.

If you run the test suite directly you can pass the `--overwrite` flag to tell
the test suite to overwrite the reference.  Because the reference output depends
on the system word size, you'll need to generate reference output for 32-bit and
64-bit builds of Julia to get Finch to pass tests. The easiest way to do this is
to run each 32-bit or 64-bit build of Julia on a system that supports it. You
can [Download](https://julialang.org/downloads/) multiple builds yourself or use
[juliaup](https://github.com/JuliaLang/juliaup) to manage multiple versions.
Using juliaup, it might look like this:

```
julia +release~x86 test/runtests.jl --overwrite
julia +release~x64 test/runtests.jl --overwrite
```

The test suite takes a while to run. You can filter to only run a selection of
test suites by specifying them as positional arguments, e.g.

```
./test/runtests.jl constructors conversions representation
```

This information is summarized with `./test/runtests.jl --help`

### Python test suite

[finch-tensor-python](https://github.com/finch-tensor/finch-tensor-python) contains
a separate Array API compatible test suite written in Python. It requires
Python 3.10 or later and Poetry installed.

It can be run with:

```sh
git clone https://github.com/finch-tensor/finch-tensor-python.git
cd finch-tensor-python
poetry install --with test
FINCH_REPO_PATH=<PATH_TO_FINCH_REPO> poetry run pytest tests/
```

## Benchmarking

The Finch test suite includes a benchmarking script that measures Finch
performance on a variety of kernels. It also includes some scripts to help
compare Finch performance on the feature branch to the main branch. To run the
benchmarking script, run `./benchmarks/runbenchmarks.jl`. To run the comparison
script, run `./benchmarks/runjudge.jl`. Both scripts take a while to run and
generate a report at the end.

## Documentation

The `/docs` directory includes Finch documentation in `/src`, and a built
website in `/build`. You can build the website with `./docs/make.jl`. You can
run doctests with `./docs/test.jl`, and fix doctests with `./docs/fix.jl`,
though both are included as part of the test suite.
