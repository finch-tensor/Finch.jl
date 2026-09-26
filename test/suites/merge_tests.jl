@testitem "merges" begin
    using Base.Iterators
    using Finch: Structure
    #TODO this is a hack to get around the fact that we don't call leaf_instance on interpolated values
    #and leaf_instance isn't super robust
    using Finch.FinchNotation: literal_instance
    fmts = [
        (;
            fmt=(z) -> Tensor(Dense(SparseList(Element(z)))),
            proto=[literal_instance(walk), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseList(Element(z)))),
            proto=[literal_instance(gallop), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseList(Element(z)))),
            proto=[literal_instance(follow), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseBlockList(Element(z)))),
            proto=[literal_instance(walk), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseBlockList(Element(z)))),
            proto=[literal_instance(gallop), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseByteMap(Element(z)))),
            proto=[literal_instance(walk), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseByteMap(Element(z)))),
            proto=[literal_instance(gallop), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseDict(Element(z)))),
            proto=[literal_instance(walk), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseDict(Element(z)))),
            proto=[literal_instance(follow), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseCOO{1}(Element(z)))),
            proto=[literal_instance(walk), literal_instance(follow)],
        ),
        (;
            fmt=(z) -> Tensor(SparseCOO{2}(Element(z))),
            proto=[literal_instance(walk), literal_instance(walk)],
        ),
        (;
            fmt=(z) -> Tensor(Dense(SparseRunList(Element(z)))),
            proto=[literal_instance(walk), literal_instance(follow)],
        ),
    ]

    dtss = [
        (; fill_value=0.0, data=fill(0, 5, 5)),
        (; fill_value=0.0, data=fill(1, 5, 5)),
        (;
            fill_value=0.0,
            data=[
                0.0 0.1 0.0 0.0 0.0;
                0.0 0.8 0.0 0.0 0.0;
                0.0 0.2 0.1 0.0 0.0;
                0.4 0.0 0.3 0.5 0.2;
                0.0 0.4 0.8 0.1 0.5],
        ),
        (;
            fill_value=0.0,
            data=[
                0.0 0.0 0.0 0.0 0.0;
                0.0 0.0 0.0 0.0 0.0;
                0.0 0.0 0.0 0.0 0.0;
                0.0 0.0 0.0 0.0 0.0;
                0.0 0.4 0.0 0.0 0.0],
        ),
        (;
            fill_value=0.0,
            data=[
                0.0 0.0 0.0 0.0 0.0;
                0.2 0.2 0.0 0.0 0.0;
                0.0 0.0 0.2 0.7 0.0;
                0.0 0.0 0.0 0.0 0.1;
                0.0 0.0 0.0 0.0 0.0],
        ),
    ]

    @testset "diagmask" begin
        for fmt in fmts
            @testset "$(summary(fmt.fmt(0.0)))[$(fmt.proto[1]), $(fmt.proto[2])]" begin
                for dts in dtss
                    a = dropfills!(fmt.fmt(dts.fill_value), dts.data)
                    b = Tensor(SparseCOO{2}(Element(dts.fill_value)))

                    @finch (
                        b .= 0;
                        for j in _, i in _
                            b[i, j] =
                                a[$(fmt.proto[1])(i), $(fmt.proto[2])(j)] * diagmask[i, j]
                        end
                    )

                    refdata = [
                        dts.data[i, j] * (j == i) for (i, j) in product(axes(dts.data)...)
                    ]
                    ref = dropfills!(Tensor(SparseCOO{2}(Element(dts.fill_value))), refdata)
                    @test Structure(b) == Structure(ref)
                end
            end
        end
    end

    @testset "lotrimask" begin
        for fmt in fmts
            @testset "$(summary(fmt.fmt(0.0)))[$(fmt.proto[1]), $(fmt.proto[2])]" begin
                for dts in dtss
                    a = dropfills!(fmt.fmt(dts.fill_value), dts.data)
                    b = Tensor(SparseCOO{2}(Element(dts.fill_value)))
                    @finch (
                        b .= 0;
                        for j in _, i in _
                            b[i, j] =
                                a[$(fmt.proto[1])(i), $(fmt.proto[2])(j)] * lotrimask[i, j]
                        end
                    )
                    refdata = [
                        dts.data[i, j] * (j <= i) for (i, j) in product(axes(dts.data)...)
                    ]
                    ref = dropfills!(Tensor(SparseCOO{2}(Element(dts.fill_value))), refdata)
                    @test Structure(b) == Structure(ref)
                end
            end
        end
    end

    @testset "uptrimask" begin
        for fmt in fmts
            @testset "$(summary(fmt.fmt(0.0)))[$(fmt.proto[1]), $(fmt.proto[2])]" begin
                for dts in dtss
                    a = dropfills!(fmt.fmt(dts.fill_value), dts.data)
                    b = Tensor(SparseCOO{2}(Element(dts.fill_value)))
                    @finch (
                        b .= 0;
                        for j in _, i in _
                            b[i, j] =
                                a[$(fmt.proto[1])(i), $(fmt.proto[2])(j)] * uptrimask[i, j]
                        end
                    )
                    refdata = [
                        dts.data[i, j] * (j >= i) for (i, j) in product(axes(dts.data)...)
                    ]
                    ref = dropfills!(Tensor(SparseCOO{2}(Element(dts.fill_value))), refdata)
                    @test Structure(b) == Structure(ref)
                end
            end
        end
    end

    #=
    @testset "bandmask" begin
        for fmt in fmts
            @testset "$(summary(fmt.fmt(0.0)))[$(fmt.proto[1]), $(fmt.proto[2])]" begin
                for dts in dtss
                    a = dropfills!(fmt.fmt(dts.fill_value), dts.data)
                    b = Tensor(SparseCOO{2}(Element(dts.fill_value)))
                    @finch (b .= 0; for j=_, i=_; b[i, j] = a[$(fmt.proto[1])(i), $(fmt.proto[2])(j)] * bandmask[i, j - 1, j + 1] end)
                    refdata = [dts.data[i, j] * (j - 1 <= i <= j + 1) for (i, j) in product(axes(dts.data)...)]
                    ref = dropfills!(Tensor(SparseCOO{2}(Element(dts.fill_value))), refdata)
                    @test Structure(b) == Structure(ref)
                end
            end
        end
    end
    =#

    @testset "plus times" begin
        n = 0
        for a_fmt in fmts
            for b_fmt in fmts[1:2]
                a_str = "$(summary(a_fmt.fmt(0.0)))[$(a_fmt.proto[1]), $(a_fmt.proto[2])]"
                b_str = "$(summary(b_fmt.fmt(0.0)))[$(b_fmt.proto[1]), $(b_fmt.proto[2])]"
                @testset "+* $a_str $b_str" begin
                    for a_dts in dtss
                        for b_dts in dtss
                            a = dropfills!(a_fmt.fmt(a_dts.fill_value), a_dts.data)
                            b = dropfills!(b_fmt.fmt(b_dts.fill_value), b_dts.data)
                            c = Tensor(SparseCOO{2}(Element(a_dts.fill_value)))
                            d = Tensor(SparseCOO{2}(Element(a_dts.fill_value)))
                            @finch (
                                c .= 0;
                                for j in _, i in _
                                    c[i, j] =
                                        a[$(a_fmt.proto[1])(i), $(a_fmt.proto[2])(j)] +
                                        b[$(b_fmt.proto[1])(i), $(b_fmt.proto[2])(j)]
                                end
                            )
                            @finch (
                                d .= 0;
                                for j in _, i in _
                                    d[i, j] =
                                        a[$(a_fmt.proto[1])(i), $(a_fmt.proto[2])(j)] *
                                        b[$(b_fmt.proto[1])(i), $(b_fmt.proto[2])(j)]
                                end
                            )
                            c_ref = dropfills!(
                                Tensor(SparseCOO{2}(Element(a_dts.fill_value))),
                                a_dts.data .+ b_dts.data,
                            )
                            d_ref = dropfills!(
                                Tensor(SparseCOO{2}(Element(a_dts.fill_value))),
                                a_dts.data .* b_dts.data,
                            )
                            @test Structure(c) == Structure(c_ref)
                            @test Structure(d) == Structure(d_ref)
                        end
                    end
                end
            end
        end
    end
end

@testitem "sparse_diagmask" begin
    let
        data = [
            1 4 7 0 0 12
            2 5 0 9 0 0
            0 6 0 0 0 13
            3 0 8 10 0 0
            0 0 0 11 0 14
        ]
        a = dropfills!(Tensor(Dense(SparseList(Element(0)))), data)
        b = Tensor(Dense(SparseList(Element(0))))
        m, n = size(data)
        for k in (-7, -2, 0, 1, 7)
            diagonal = Finch.offset(diagmask, 0, -k)
            for mask in
                (diagonal, Finch.window(diagonal, Finch.Extent(1, m), Finch.Extent(1, n)))
                @finch begin
                    b .= 0
                    for j in _, i in _
                        b[i, j] = a[i, j] * mask[i, j]
                    end
                end
                @test Array(b) == [data[i, j] * (i == j - k) for i in 1:m, j in 1:n]
            end
        end
    end
end
