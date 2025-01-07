# This file tests the ground-truth level representations through conversion of
# tensors between different representations, the conversion of tensors to and
# from their string representation, validation against reference getindex, and
# validation against reference conversion code

@testset "representation" begin
    @info "Testing Tensor Representation"

    modifier_levels = [
        (key = "Separate", Lvl = Separate),
        (key = "Mutex", Lvl = Mutex, repr = false),
    ]

    basic_levels = [
        (key = "Dense", Lvl = Dense, pattern = false),
        (key = "RunList", Lvl = RunList, pattern = false),
        (key = "RunListLazy", Lvl = (base) -> RunList(base; merge = false), pattern = false),
        (key = "SparseList", Lvl = SparseList),
        (key = "SparseBlockList", Lvl = SparseBlockList),
        (key = "SparseBand", Lvl = SparseBand, pattern = false),
        (key = "SparseByteMap", Lvl = SparseByteMap),
        (key = "SparseRunList", Lvl = SparseRunList),
        (key = "SparseRunListLazy", Lvl = (base) -> SparseRunList(base, merge=false)),
        (key = "SparseDict", Lvl = SparseDict),
        (key = "SparsePoint", Lvl = SparsePoint, filter = (key) -> key in ["6x_one_bool"]),
        (key = "SparseInterval", Lvl = SparseInterval, filter = (key) -> key in ["6x_one_bool"]),
        (key = "SparseList{Separate}", Lvl = (base) -> SparseList(Separate(base))),
    ]

    multi_levels = [
        (key = "SparseCOO", Lvl = SparseCOO),
    ]

    levels_1D = []
    levels_2D = []

    for lvl in basic_levels
        lvl = merge((filter = (x) -> true,), lvl)
        push!(levels_1D, lvl)
        push!(levels_2D, merge(lvl, (key = "$(lvl.key){SparseList}", Lvl = (base) -> lvl.Lvl(SparseList(base)), filter = (key) -> lvl.filter("$(key)_sparse_inner"))))
        push!(levels_2D, merge(lvl, (key = "$(lvl.key){Dense}", Lvl = (base) -> lvl.Lvl(Dense(base)), filter = (key) -> lvl.filter("$(key)_dense_inner"), pattern=false)))
        push!(levels_2D, merge(lvl, (key = "SparseList{$(lvl.key)}", Lvl = (base) -> SparseList(lvl.Lvl(base)), filter = (key) -> lvl.filter("$(key)_sparse_outer"))))
        push!(levels_2D, merge(lvl, (key = "Dense{$(lvl.key)}", Lvl = (base) -> Dense(lvl.Lvl(base)), filter = (key) -> lvl.filter("$(key)_dense_outer"))))
    end

    for lvl in multi_levels
        push!(levels_1D, (key = "$(lvl.key){1}", Lvl = lvl.Lvl{1}))
        push!(levels_2D, (key = "$(lvl.key){2}", Lvl = lvl.Lvl{2}))
    end

    for lvl in modifier_levels
        lvl = merge((filter = (x) -> true, repr = true), lvl)
        push!(levels_1D, (key = "Dense{$(lvl.key)}", Lvl = (base) -> lvl.Lvl(Dense(base)), filter = (key) -> lvl.filter("$(key)_dense"), pattern=false, repr = lvl.repr))
    end

    ios = Dict()
    compiled = Set()

    for (key, arr) = [
        ("5x_false", fill(false, 5)),
        ("5x_true", fill(true, 5)),
        ("6x_bool_mix", [false, true, true, false, false, true]),
        ("6x_one_bool", [false, false, true, false, false, false]),
        ("1111x_bool_mix", begin
            x = fill(false, 1111)
            x[2] = true
            x[3]= true
            x[555:999] .= true
            x[1001] = true
            x
        end),
        ("11x_bool_mix", begin
            x = fill(false, 11)
            x[2] = true
            x[3]= true
            x[5:9] .= true
            x[11] = true
            x
        end),
        ("6x_float_mix", [0.0, 2.0, 2.0, 0.0, 3.0, 3.0]),
        ("4x_zeros", [0.0, 0.0, 0.0, 0.0]),
        ("5x_zeros", fill(0.0, 5)),
        ("5x_ones", fill(1.0, 5)),
        ("9x_float_mix", [0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 3.0, 0.0]),
        ("1111x_float_mix", begin
            x = zeros(1111)
            x[2] = 20.0
            x[3]=30.0
            x[555]=5550.0
            x[666]=6660.0
            x
        end),

    ]
        for lvl in levels_1D
            lvl = merge((filter = (x) -> true, pattern = true, repr = true), lvl)
            if lvl.filter(key)
                leaf = () -> Element{fill_value(arr), eltype(arr)}()
                ref = Tensor(SparseList(leaf()))
                res = Tensor(SparseList(leaf()))
                ref = dropfills!(ref, arr)
                tmp = Tensor(lvl.Lvl(leaf()))
                @testset "convert $(key) $(lvl.key)(Element())" begin
                    fname = "representation/convert_to_$(lvl.key){Element{$(fill_value(arr))}}.jl"
                    get!(ios, fname) do
                        io = IOBuffer()
                        show(io, @finch_code (tmp .= 0; for i=_; tmp[i] = ref[i] end))
                        io
                    end
                    fname = "representation/convert_from_$(lvl.key){Element{$(fill_value(arr))}}.jl"
                    get!(ios, fname) do
                        io = IOBuffer()
                        show(io, @finch_code (res .= 0; for i=_; res[i] = tmp[i] end))
                        io
                    end
                    roundtrip_key = typeof((res, tmp, ref))
                    if !(roundtrip_key in compiled)
                        eval(@finch_kernel function roundtrip(res, tmp, ref)
                            tmp .= 0
                            for i = _
                                tmp[i] = ref[i]
                            end
                            res .= 0
                            for i = _
                                res[i] = tmp[i]
                            end
                            return res
                        end)
                        push!(compiled, roundtrip_key)
                    end
                    res = roundtrip(res, tmp, ref).res
                    @test size(res) == size(ref)
                    @test axes(res) == axes(ref)
                    @test ndims(res) == ndims(ref)
                    @test eltype(res) == eltype(ref)
                    @test res == ref
                    @test isequal(res, ref)
                    @finch (tmp .= 0; for i=_; tmp[i] = ref[i] end)
                    @finch (res .= 0; for i=_; res[i] = tmp[i] end)
                    @test size(res) == size(ref)
                    @test axes(res) == axes(ref)
                    @test ndims(res) == ndims(ref)
                    @test eltype(res) == eltype(ref)
                    @test res == ref
                    @test isequal(res, ref)
                    if lvl.pattern
                        @test Structure(ref) == Structure(res)
                    end

                    tmp = dropfills!(tmp, arr)
                    fname = "representation/$(lvl.key)_representation.txt"
                    io = get!(ios, fname) do
                        io = IOBuffer()
                        println(io, "$(lvl.key) representation:")
                        println(io)
                        io
                    end
                    println(io, "$key: ", arr)
                    if lvl.repr
                        @test Structure(tmp) == Structure(eval(Meta.parse(repr(tmp))))
                    end
                    @test reference_isequal(tmp, arr)
                    @test Finch.AsArray(tmp) == arr
                    println(io, "tensor: ", repr(tmp))
                    println(io, "countstored: ", countstored(tmp))
                end
            end
        end
    end

    for (key, arr) in [
        ("5x5_falses", fill(false, 5, 5)),
        ("5x5_trues", fill(true, 5, 5)),
        ("4x4_one_bool",
            [false false  false true ;
            false false false false
            true  false false false
            false true  false false ]),
        ("5x4_bool_mix",
            [false true  false true ;
            false false false false
            true  true  true  true
            true  true  true  true
            false true  false true ]),
        ("5x5_zeros", fill(0.0, 5, 5)),
        ("5x5_ones", fill(1.0, 5, 5)),
        ("5x5_float_mix",
            [0.0 1.0 2.0 2.0 3.0 ;
            0.0 0.0 0.0 0.0 0.0 ;
            1.0 1.0 2.0 0.0 0.0 ;
            0.0 0.0 0.0 3.0 0.0 ;
            0.0 0.0 0.0 0.0 0.0]),
    ]
        for lvl in levels_2D
            lvl = merge((filter = (x) -> true, pattern = true, repr = true), lvl)
            if lvl.filter(key)
                leaf = () -> Element{fill_value(arr), eltype(arr)}()
                ref = Tensor(SparseList(SparseList(leaf())))
                res = Tensor(SparseList(SparseList(leaf())))
                ref = dropfills!(ref, arr)
                tmp = Tensor(lvl.Lvl(leaf()))
                @testset "convert $(key) $(lvl.key)(Element())" begin
                    roundtrip_key = typeof((res, tmp, ref))
                    if !(roundtrip_key in compiled)
                        eval(@finch_kernel function roundtrip(res, tmp, ref)
                            tmp .= 0
                            for j=_, i=_
                                tmp[i, j] = ref[i, j]
                            end
                            res .= 0
                            for j=_, i=_
                                res[i, j] = tmp[i, j]
                            end
                            return res
                        end)
                        push!(compiled, roundtrip_key)
                    end
                    res = roundtrip(res, tmp, ref).res
                    @test size(res) == size(ref)
                    @test axes(res) == axes(ref)
                    @test ndims(res) == ndims(ref)
                    @test eltype(res) == eltype(ref)
                    @test res == ref
                    @test isequal(res, ref)
                    @finch (tmp .= 0; for j=_, i=_; tmp[i, j] = ref[i, j] end)
                    @finch (res .= 0; for j=_, i=_; res[i, j] = tmp[i, j] end)
                    @test size(tmp) == size(ref)
                    @test axes(res) == axes(ref)
                    @test ndims(tmp) == ndims(ref)
                    @test eltype(tmp) == eltype(ref)
                    @test res == ref
                    @test isequal(res, ref)
                    if lvl.pattern
                        @test Structure(ref) == Structure(res)
                    end
                    fname = "representation/$(lvl.key)_representation.txt"
                    io = get!(ios, fname) do
                        io = IOBuffer()
                        println(io, "$(lvl.key) representation:")
                        println(io)
                        io
                    end
                    println(io, "$key: ", arr)
                    if lvl.repr
                        @test Structure(tmp) == Structure(eval(Meta.parse(repr(tmp))))
                    end
                    @test reference_isequal(tmp, arr)
                    println(io, "tensor: ", repr(tmp))
                    println(io, "countstored: ", countstored(tmp))
                end
            end
        end
    end

    for (fname, io) in ios
        @testset "$fname" begin
            @test check_output(fname, String(take!(io)))
        end
    end

end
