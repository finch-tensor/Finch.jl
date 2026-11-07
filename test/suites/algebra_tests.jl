
@testitem "algebra" begin
    struct MyAlgebra <: Finch.AbstractAlgebra end
    struct MyAlgebra2 <: Finch.AbstractAlgebra end

    u = Tensor(SparseList(Element(1)), [3, 1, 6, 1, 9, 1, 4, 1, 8, 1])
    v = Tensor(SparseList(Element(1)), [1, 2, 3, 1, 1, 1, 1, 4, 1, 1])
    w = Tensor(SparseList(Element(1)))

    @finch (w.=1;
        for i in _
            w[i] = gcd(u[i], v[i])
        end)

    @test pattern!(w) == [1, 1, 1, 0, 1, 0, 1, 1, 1, 0]

    Finch.isassociative(::MyAlgebra, ::typeof(gcd)) = true
    Finch.iscommutative(::MyAlgebra, ::typeof(gcd)) = true
    Finch.isannihilator(::MyAlgebra, ::typeof(gcd), x) = x == 1

    @finch algebra = MyAlgebra() (w.=1;
        for i in _
            w[i] = gcd(u[i], v[i])
        end)

    @test pattern!(w) == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    @finch algebra = MyAlgebra2() (w.=1;
        for i in _
            w[i] = gcd(u[i], v[i])
        end)

    @test pattern!(w) == [1, 1, 1, 0, 1, 0, 1, 1, 1, 0]

    Finch.isassociative(::MyAlgebra2, ::typeof(gcd)) = true
    Finch.iscommutative(::MyAlgebra2, ::typeof(gcd)) = true
    Finch.isannihilator(::MyAlgebra2, ::typeof(gcd), x) = x == 1

    @finch algebra = MyAlgebra2() (w.=1;
        for i in _
            w[i] = gcd(u[i], v[i])
        end)

    @test pattern!(w) == [1, 1, 1, 0, 1, 0, 1, 1, 1, 0]

    Finch.refresh()

    @finch algebra = MyAlgebra2() (w.=1;
        for i in _
            w[i] = gcd(u[i], v[i])
        end)

    @test pattern!(w) == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
end

@testitem "logical" begin
    @test Finch.logical_and(2, 0.5, true) === true
    @test Finch.logical_and(2, 0) === false
    @test Finch.logical_or(0, 0.0, false) === false
    @test Finch.logical_or(0, NaN) === true
    @test Finch.logical_xor(2, 0.5) === false
    @test Finch.logical_xor(2, 0) === true

    a = Tensor(SparseList(Element(0)), [0, 5, 0, 0, 3, 0, 0, 0])
    b = Tensor(SparseList(Element(0)), [0, 0, 0, 0, 7, 0, 2, 0])
    c = Tensor(SparseList(Element(false)))

    # Only the intersection is visited, so no false is stored.
    @finch (c.=false;
        for i in _
            c[i] = Finch.logical_and(a[i], b[i])
        end)
    @test c == [0, 0, 0, 0, 1, 0, 0, 0]
    @test pattern!(c) == [0, 0, 0, 0, 1, 0, 0, 0]

    @finch (c.=false;
        for i in _
            c[i] = Finch.logical_or(a[i], b[i])
        end)
    @test c == [0, 1, 0, 0, 1, 0, 1, 0]
    @test pattern!(c) == [0, 1, 0, 0, 1, 0, 1, 0]

    @finch (c.=false;
        for i in _
            c[i] = Finch.logical_xor(a[i], b[i])
        end)
    @test c == [0, 1, 0, 0, 0, 0, 1, 0]

    # Where t is at its fill, logical_and(a[i], true) is still a Bool.
    t = Tensor(SparseList(Element(true)), [true, false, true, true, true, true, true, true])
    @finch (c.=false;
        for i in _
            c[i] = Finch.logical_and(a[i], t[i])
        end)
    @test c == [0, 0, 0, 0, 1, 0, 0, 0]

    x = Scalar(false)
    @finch (x.=false;
        for i in _
            x[] << Finch.logical_or >>= a[i]
        end)
    @test x[] === true
    @finch (x.=true;
        for i in _
            x[] << Finch.logical_and >>= a[i]
        end)
    @test x[] === false
    @finch (x.=true;
        for i in _
            x[] << Finch.logical_and >>= t[i]
        end)
    @test x[] === false
end

@testitem "scaled norms" begin
    using LinearAlgebra

    function scaled_norm(a, p)
        if p == 2
            x = Scalar((0.0, 0.0))
            @finch for i in _
                x[] << Finch.add_scaled_square >>= Finch.scaled_square(abs(a[i]))
            end
            return Finch.root_scaled_square(x[])
        elseif p > 0
            x = Scalar((0.0, 0.0))
            @finch for i in _
                x[] << Finch.add_scaled_power(p) >>= Finch.scaled_power(p)(abs(a[i]))
            end
            return Finch.root_scaled_power(p)(x[])
        else
            x = Scalar((0.0, Inf))
            @finch for i in _
                x[] << Finch.add_scaled_negative_power(p) >>= Finch.scaled_negative_power(p)(
                    abs(a[i])
                )
            end
            return Finch.root_scaled_negative_power(p)(x[])
        end
    end

    for v in (
        [0, 3.0, 0, 0, -4.0, 0, 1.5, 0],
        [2.0, -3.0, 0.5, 4.0],
        [0, 3e200, 0, -4e200],
        [3e-200, 4e-200],
        zeros(4),
    )
        tns = Tensor(SparseList(Element(0.0)), v)
        for p in (2, 3.0, 0.5, -1.0, -2.0)
            @test scaled_norm(tns, p) ≈ norm(v, p)
        end
    end

    @test isnan(scaled_norm(Tensor(Dense(Element(0.0)), [1.0, NaN, 2.0]), 2))
    @test Finch.scaled_square(3) === (1.0, 3)
    @test Finch.scaled_negative_power(-1.0)(0.0f0) === (Inf32, 0.0f0)
end
