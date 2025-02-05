#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
    Pkg.instantiate()
end

using Documenter
using Documenter.Remotes
using Finch

DocMeta.setdocmeta!(
    Finch, :DocTestSetup, :(using Finch; using SparseArrays); recursive=true
)

makedocs(;
    modules=[Finch],
    authors="Willow Ahrens",
    repo=Remotes.GitHub("finch-tensor", "Finch.jl"),
    sitename="Finch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://finch-tensor.github.io/Finch.jl",
        assets=["assets/favicon.ico"],
        size_threshold=1_000_000,
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Documentation" => [
            "Tensor Formats" => "docs/tensor_formats.md",
            "High-Level Array API" => "docs/array_api.md",
            "Sparse and Structured Utilities" => "docs/sparse_utils.md",
            "User-Defined Functions" => "docs/user-defined_functions.md",
            "FileIO" => "docs/fileio.md",
            "Advanced: Finch Language" => [
                "Calling Finch" => "docs/language/calling_finch.md",
                "The Finch Language" => "docs/language/finch_language.md",
                "Dimensionalization" => "docs/language/dimensionalization.md",
                #"Tensor Lifecycles" => "docs/language/tensor_lifecycles.md",
                "Index Sugar" => "docs/language/index_sugar.md",
                "Mask Sugar" => "docs/language/mask_sugar.md",
                "Iteration Protocols" => "docs/language/iteration_protocols.md",
                "Parallelization" => "docs/language/parallelization.md",
                "Interoperability" => "docs/language/interoperability.md",
                "Optimization Tips" => "docs/language/optimization_tips.md",
                "Benchmarking Tips" => "docs/language/benchmarking_tips.md",
                #"Debugging Tips" => "docs/language/debugging_tips.md",
            ],
            "Developers: Internal Details" => [
                #"Staging" => "docs/internals/staging.md",
                "Virtualization" => "docs/internals/virtualization.md",
                "Tensor Interface" => "docs/internals/tensor_interface.md",
                "Compiler Interfaces" => "docs/internals/compiler_interface.md",
                "Finch Notation" => "docs/internals/finch_notation.md",
                "Finch Logic" => "docs/internals/finch_logic.md",
                #        "Looplets and Coiteration" => "internals/looplets_coiteration.md",
            ],
        ],
        "Community and Contributions" => "CONTRIBUTING.md",
        "Appendices and Additional Resources" => [
            #"Glossary" => "appendices/glossary.md",
            #"FAQs" => "appendices/faqs.md",
            "Directory Structure" => "appendices/directory_structure.md",
            "Code Listing" => "appendices/directory_structure.md",
            #"Changelog" => "appendices/changelog.md",
            #"Publications and Articles" => "appendices/publications_articles.md",
        ],
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/finch-tensor/Finch.jl",
    devbranch="main",
)
