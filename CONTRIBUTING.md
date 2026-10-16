We welcome contributions to Galley, and follow the [Julia contributing
guidelines](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md).  If
you use or want to use Galley and have a question or bug, please do file a
[Github issue](https://github.com/finch-tensor/FinchGalley.jl/issues)!  If you want
to contribute to Galley, please first file an issue to double check that there is
interest from a contributor in the feature.

## Versions

The Galley API is stable and
we follow [semantic
versioning](https://semver.org/) conventions.
The main branch of the Galley repo is the most up-to-date development branch.
While it is not stable, it should always pass tests.

Contributors will develop and test Galley from a local directory. Please see the
[Package documentation](https://pkgdocs.julialang.org/v1/getting-started/) for more
info, particularly the section on [developing](https://pkgdocs.julialang.org/v1/managing-packages/#developing).

To determine which version of
Galley you have, run `Pkg.status("Galley")` in the Julia REPL. If the installed
version of Galley tracks a local path, the output will include the path like so:

```
Status `~/.julia/environments/v1.9/Project.toml`
  [9177782c] Galley v0.5.4 `~/Projects/Galley.jl`
```

If the installed version of Galley tracks a particular version (probably not what
you want since it will not reflect local changes), the output will look like this:

```
Status `~/.julia/environments/v1.8/Project.toml`
  [9177782c] Galley v0.5.4
```

## Utilities

Galley include several scripts that can be executed directly, e.g. `runtests.jl`.
These scripts are all have local [Pkg
environments](https://pkgdocs.julialang.org/v1/getting-started/#Getting-Started-with-Environments).
The scripts include convenience headers to automatically use their respective
environments, so you won't need to worry about `--project=.` flags, etc.

## Testing

All pull requests should pass continuous integration testing before merging.
The test suite has a few options, which are accessible through running the test
suite directly as `./test/runtests.jl`.

Galley compares compiler output against reference versions.

If you have the appropriate permissions, you can run the
[FixBot](https://github.com/finch-tensor/FinchGalley.jl/actions/workflows/FixBot.yml)
github action on your PR branch to automatically generate output for both 32-bit
and 64-bit builds.

If you run the test suite directly you can pass the `--overwrite` flag to tell
the test suite to overwrite the reference.  Because the reference output depends
on the system word size, you'll need to generate reference output for 32-bit and
64-bit builds of Julia to get Galley to pass tests. The easiest way to do this is
to run each 32-bit or 64-bit build of Julia on a system that supports it. You
can [Download](https://julialang.org/downloads/) multiple builds yourself or use
[juliaup](https://github.com/JuliaLang/juliaup) to manage multiple versions.
Using juliaup, it might look like this:

```
julia +release~x86 test/runtests.jl --overwrite
julia +release~x64 test/runtests.jl --overwrite
```

The test suite takes a while to run. You can parallelize by setting the `-p`
option to configure the number of processors.

```
./test/runtests.jl -p 22
```

You can also filter to only run a selection of test suitesusing the `--include` or
`--exclude` arguments, or

```
./test/runtests.jl --include constructors interface_einsum interface_asmd
```

This information is summarized with `./test/runtests.jl --help`

### Python test suite

[galley-jl-python](https://github.com/finch-tensor/galley-jl-python), formerly
finch-tensor-python, contains a separate Array API compatible test suite written
in Python. To match the `galley-jl-python-tests` job in `.github/workflows/CI.yml`,
use release `v0.2.13`, Julia 1.10, Python 3.11, and Poetry.

With Julia 1.10 on your `PATH`, run the following commands, replacing
`/absolute/path/to/Galley.jl` with the path to your Galley.jl checkout:

```sh
git clone --branch v0.2.13 https://github.com/finch-tensor/galley-jl-python.git finch-tensor
python3 /absolute/path/to/Galley.jl/test/rename_python_backend.py finch-tensor
cd finch-tensor
poetry env use python3.11
poetry install --extras test
poetry run python develop.py --path "/absolute/path/to/Galley.jl"
poetry run pytest tests/
```

The adapter updates the wrapper's Julia dependency and module references to
`Galley` while retaining compiler API names such as `FinchNotation` and `@finch`.

## Benchmarking

The Galley test suite includes a benchmarking script that measures Galley
performance on a variety of kernels. It also includes some scripts to help
compare Galley performance on the feature branch to the main branch. To run the
benchmarking script, run `./benchmarks/runbenchmarks.jl`. To run the comparison
script, run `./benchmarks/runjudge.jl`. Both scripts take a while to run and
generate a report at the end.

## Documentation

The `/docs` directory includes Galley documentation in `/src`, and a built
website in `/build`. You can build the website with `./docs/make.jl`. You can
run doctests with `./docs/test.jl`, and fix doctests with `./docs/fix.jl`,
though both are included as part of the test suite.

## Code Style

We use [Blue Style](https://github.com/JuliaDiff/BlueStyle) formatting, with a few tweaks
defined in `.JuliaFormatter.toml`. Running the tests in overwrite mode will
automatically reformat your code, but you can also add [`JuliaFormatter`](https://domluna.github.io/JuliaFormatter.jl/stable/#Editor-Plugins) to your editor to reformat as you go.
