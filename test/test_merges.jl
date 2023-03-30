@testset "merges" begin
    @info "Testing Merge Kernels"
    using Base.Iterators

    fmts = [
        (;fmt = (z) -> @fiber(d(sl(e(z)))), proto = [walk, follow]),
        (;fmt = (z) -> @fiber(d(sl(e(z)))), proto = [gallop, follow]),
        (;fmt = (z) -> @fiber(d(svb(e(z)))), proto = [walk, follow]),
        (;fmt = (z) -> @fiber(d(svb(e(z)))), proto = [gallop, follow]),
        (;fmt = (z) -> @fiber(d(sbm(e(z)))), proto = [walk, follow]),
        (;fmt = (z) -> @fiber(d(sbm(e(z)))), proto = [gallop, follow]),
        (;fmt = (z) -> @fiber(d(sc{1}(e(z)))), proto = [walk, follow]),
        (;fmt = (z) -> @fiber(sc{2}(e(z))), proto = [walk, walk]),
        (;fmt = (z) -> @fiber(d(sh{1}(e(z)))), proto = [walk,follow]),
        (;fmt = (z) -> @fiber(sh{2}(e(z))), proto = [walk, walk]),
        #(;fmt = (z) -> @fiber(d(rl(z))), proto = [walk, follow]),
    ]

    dtss = [
        (;default = 0.0, data = fill(0, 5, 5), ),
        (;default = 0.0, data = fill(1, 5, 5), ),
        (;default = 0.0, data = [
            0.0 0.1 0.0 0.0 0.0;
            0.0 0.8 0.0 0.0 0.0;
            0.0 0.2 0.1 0.0 0.0;
            0.4 0.0 0.3 0.5 0.2;
            0.0 0.4 0.8 0.1 0.5],),
        (;default = 0.0, data = [
            0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0;
            0.0 0.4 0.0 0.0 0.0],),
        (;default = 0.0, data = [
            0.0 0.0 0.0 0.0 0.0;
            0.2 0.2 0.0 0.0 0.0;
            0.0 0.0 0.2 0.7 0.0;
            0.0 0.0 0.0 0.0 0.1;
            0.0 0.0 0.0 0.0 0.0],),
    ]

    @testset "diagmask" begin
        for fmt in fmts
            for dts in dtss
                a = dropdefaults!(fmt.fmt(dts.default), dts.data)
                @testset "$(summary(a))[::$(fmt.proto[1]), ::$(fmt.proto[2])]" begin
                    b = @fiber(sc{2}(e(dts.default)))
                    @finch (b .= 0; @loop j i b[i, j] = a[i::(fmt.proto[1]), j::(fmt.proto[2])] * diagmask[i, j])
                    refdata = [dts.data[i, j] * (j == i) for (i, j) in product(axes(dts.data)...)]
                    ref = dropdefaults!(@fiber(sc{2}(e(dts.default))), refdata)
                    @test isstructequal(b, ref)
                end
            end
        end
    end

    @testset "lotrimask" begin
        for fmt in fmts
            for dts in dtss
                a = dropdefaults!(fmt.fmt(dts.default), dts.data)
                @testset "$(summary(a))[::$(fmt.proto[1]), ::$(fmt.proto[2])]" begin
                    b = @fiber(sc{2}(e(dts.default)))
                    @finch (b .= 0; @loop j i b[i, j] = a[i::(fmt.proto[1]), j::(fmt.proto[2])] * lotrimask[i, j])
                    refdata = [dts.data[i, j] * (j <= i) for (i, j) in product(axes(dts.data)...)]
                    ref = dropdefaults!(@fiber(sc{2}(e(dts.default))), refdata)
                    @test isstructequal(b, ref)
                end
            end
        end
    end

    @testset "uptrimask" begin
        for fmt in fmts
            for dts in dtss
                a = dropdefaults!(fmt.fmt(dts.default), dts.data)
                @testset "$(summary(a))[::$(fmt.proto[1]), ::$(fmt.proto[2])]" begin
                    b = @fiber(sc{2}(e(dts.default)))
                    @finch (b .= 0; @loop j i b[i, j] = a[i::(fmt.proto[1]), j::(fmt.proto[2])] * uptrimask[i, j])
                    refdata = [dts.data[i, j] * (j >= i) for (i, j) in product(axes(dts.data)...)]
                    ref = dropdefaults!(@fiber(sc{2}(e(dts.default))), refdata)
                    @test isstructequal(b, ref)
                end
            end
        end
    end

    @testset "bandmask" begin
        for fmt in fmts
            for dts in dtss
                a = dropdefaults!(fmt.fmt(dts.default), dts.data)
                @testset "$(summary(a))[::$(fmt.proto[1]), ::$(fmt.proto[2])]" begin
                    b = @fiber(sc{2}(e(dts.default)))
                    @finch (b .= 0; @loop j i b[i, j] = a[i::(fmt.proto[1]), j::(fmt.proto[2])] * bandmask[i, j - 1, j + 1])
                    refdata = [dts.data[i, j] * (j - 1 <= i <= j + 1) for (i, j) in product(axes(dts.data)...)]
                    ref = dropdefaults!(@fiber(sc{2}(e(dts.default))), refdata)
                    @test isstructequal(b, ref)
                end
            end
        end
    end

    @testset "plus times" begin
        n = 0
        for a_fmt in fmts
            for b_fmt in fmts[1:2]
                for a_dts in dtss
                    for b_dts in dtss
                        a = dropdefaults!(a_fmt.fmt(a_dts.default), a_dts.data)
                        b = dropdefaults!(b_fmt.fmt(b_dts.default), b_dts.data)
                        a_str = "$(summary(a))[::$(a_fmt.proto[1]), ::$(a_fmt.proto[2])]"
                        b_str = "$(summary(b))[::$(b_fmt.proto[1]), ::$(b_fmt.proto[2])]"
                        @testset "+* $a_str $b_str" begin
                            c = @fiber(sc{2}(e(a_dts.default)))
                            d = @fiber(sc{2}(e(a_dts.default)))
                            @finch (c .= 0; @loop j i c[i, j] = a[i::(a_fmt.proto[1]), j::(a_fmt.proto[2])] + b[i::(b_fmt.proto[1]), j::(b_fmt.proto[2])])
                            @finch (d .= 0; @loop j i d[i, j] = a[i::(a_fmt.proto[1]), j::(a_fmt.proto[2])] * b[i::(b_fmt.proto[1]), j::(b_fmt.proto[2])])
                            c_ref = dropdefaults!(@fiber(sc{2}(e(a_dts.default))), a_dts.data .+ b_dts.data)
                            d_ref = dropdefaults!(@fiber(sc{2}(e(a_dts.default))), a_dts.data .* b_dts.data)
                            @test isstructequal(c, c_ref)
                            @test isstructequal(d, d_ref)
                        end
                    end
                end
            end
        end
    end
end