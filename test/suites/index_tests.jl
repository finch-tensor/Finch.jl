@testitem "index" setup = [CheckOutput] begin
    A = Tensor(SparseList(Element(0.0)), [2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0])
    B = Scalar{0.0}()

    @test check_output("index/sieve_hl_cond.jl", @finch_code (B .= 0;
    for j in _
        if j == 1
            B[] += A[j]
        end
    end))
    @finch (B .= 0;
    for j in _
        if j == 1
            B[] += A[j]
        end
    end)

    @test B() == 2.0

    @finch (B .= 0;
    for j in _
        if j == 2
            B[] += A[j]
        end
    end)

    @test B() == 0.0

    @test check_output("index/sieve_hl_select.jl", @finch_code (
        for j in _
            if diagmask[j, 3]
                B[] += A[j]
            end
        end
    ))

    @finch (B .= 0;
    for j in _
        if diagmask[j, 3]
            B[] += A[j]
        end
    end)

    @test B() == 3.0

    @finch (B .= 0;
    for j in _
        if diagmask[j, 4]
            B[] += A[j]
        end
    end)

    @test B() == 0.0

    @test check_output("index/gather_hl.jl", @finch_code (B[] += A[5]))

    @finch (B .= 0; B[] += A[5])

    @test B() == 4.0

    @finch (B .= 0; B[] += A[6])

    @test B() == 0.0

    @testset "scansearch" begin
        for v in [
            [
                2,
                3,
                5,
                7,
                8,
                10,
                13,
                14,
                15,
                17,
                19,
                21,
                25,
                26,
                32,
                35,
                36,
                38,
                39,
                40,
                41,
                42,
                45,
                47,
                48,
                51,
                52,
                54,
                57,
                58,
                59,
                60,
                64,
                68,
                72,
                78,
                79,
                82,
                83,
                87,
                89,
                95,
                97,
                98,
                100,
            ],
            collect(-200:3:500),
        ]
            hi = length(v)
            for lo in 1:hi, i in lo:hi
                @test Finch.scansearch(v, v[i], lo, hi) == i
                @test Finch.scansearch(v, v[i] - 1, lo, hi) ==
                    ((i > lo && v[i - 1] == v[i] - 1) ? i - 1 : i)
                @test Finch.scansearch(v, v[i] + 1, lo, hi) == i + 1
            end
        end
    end

    using SparseArrays

    A_ref = sprand(10, 0.5)
    B_ref = sprand(10, 0.5)
    C_ref = vcat(A_ref, B_ref)
    A = Tensor(SparseVector{Float64,Int64}(A_ref))
    B = Tensor(SparseVector{Float64,Int64}(B_ref))
    C = Tensor(SparseList{Int64}(Element(0.0)), 20)
    @test check_output(
        "index/concat_offset_permit.jl",
        @finch_code (C .= 0;
        for i in _
            C[i] = coalesce(A[~i], B[~(i - 10)])
        end)
    )
    @finch (C .= 0;
    for i in _
        C[i] = coalesce(A[~i], B[~(i - 10)])
    end)
    @test C == C_ref

    F = Tensor(Int64[1, 1, 1, 1, 1])

    @test check_output(
        "index/sparse_conv.jl",
        @finch_code (C .= 0;
        for i in _, j in _
            C[i] += (A[i] != 0) * coalesce(A[j - i + 3], 0) * F[j]
        end)
    )
    @finch (C .= 0;
    for i in _, j in _
        C[i] += (A[i] != 0) * coalesce(A[j - i + 3], 0) * F[j]
    end)
    C_ref = zeros(10)
    for i in 1:10
        if A_ref[i] != 0
            for j in 1:5
                k = (j - (i - 3))
                if 1 <= k <= 10
                    C_ref[i] += A_ref[k]
                end
            end
        end
    end
    @test C == C_ref

    @test check_output(
        "index/sparse_window.jl", @finch_code (C .= 0;
        for i in _
            C[i] = A[(2:4)(i)]
        end)
    )
    @finch (C .= 0;
    for i in _
        C[i] = A[(2:4)(i)]
    end)
    @test C == [A(2), A(3), A(4)]

    I = 2:4
    @finch (C .= 0;
    for i in _
        C[i] = I[i]
    end)
    @test C == [2, 3, 4]

    y = Array{Any}(undef, 4)
    x = Tensor(Dense(Element(0.0)), zeros(2))
    X = Finch.permissive(x, true)

    @finch for i in _
        y[i] := X[i]
    end

    @test isequal(y, [0.0, 0.0, missing, missing])

    y = Array{Any}(undef, 4)

    @finch for i in _
        y[i] := Finch.permissive(x, true)[i]
    end

    @test isequal(y, [0.0, 0.0, missing, missing])

    y = Array{Any}(undef, 4)

    z = Finch.permissive(x, true)

    @finch begin
        for i in _
            y[i] := z[i]
        end
    end

    @test isequal(y, [0.0, 0.0, missing, missing])

    @finch begin
        for i in 1:4
            y[i] := x[~i]
        end
    end

    @test isequal(y, [0.0, 0.0, missing, missing])

    let
        io = IOBuffer()
        println(io, "chunkmask tests")

        @repl io A = Tensor(Dense(Dense(Element(0.0))), 15, 3)
        @repl io m = Finch.chunkmask(15, 5)
        @repl io @finch begin
            for i in _
                for j in _
                    A[j, i] = m[j, i]
                end
            end
        end
        @repl io AsArray(A)

        @repl io A = Tensor(Dense(Dense(Element(0.0))), 14, 3)
        @repl io m = Finch.chunkmask(14, 5)
        @repl io @finch begin
            for i in _
                for j in _
                    A[j, i] = m[j, i]
                end
            end
        end
        @repl io AsArray(A)

        @test check_output("index/chunkmask.txt", String(take!(io)))

        io = IOBuffer()
        println(io, "splitmask tests")

        @repl io A = Tensor(Dense(Dense(Element(0.0))), 15, 5)
        @repl io m = Finch.splitmask(15, 5)
        @repl io @finch begin
            for i in _
                for j in _
                    A[j, i] = m[j, i]
                end
            end
        end
        @repl io AsArray(A)

        @repl io A = Tensor(Dense(Dense(Element(0.0))), 14, 5)
        @repl io m = Finch.splitmask(14, 5)
        @repl io @finch begin
            for i in _
                for j in _
                    A[j, i] = m[j, i]
                end
            end
        end
        @repl io AsArray(A)

        @test check_output("index/splitmask.txt", String(take!(io)))
    end

    let
        x = Scalar(0.0)
        A = Tensor(Dense(Dense(Dense(Element(0.0)))),
            reshape(
                [1 3 5 2 4 6 7 9 11 8 10 12 13 15 17 14 16 18 19 21 23 20 22 24], 2, 3, 4
            ))
        @finch begin
            x .= 0
            for i in _
                for j in _
                    for k in _
                        x[] += swizzle(A, 3, 2, 1)[i, j, k]
                    end
                end
            end
        end

        @test x[] == (24 + 1) * 24 / 2

        @test check_output(
            "index/swizzle_1.txt", @finch_code begin
                for i in _
                    for j in _
                        for k in _
                            x[] += swizzle(A, 3, 2, 1)[i, j, k]
                        end
                    end
                end
            end
        )

        x = Scalar(0.0)
        A = swizzle(
            Tensor(Dense(Dense(Dense(Element(0.0)))),
                reshape(
                    [1 3 5 2 4 6 7 9 11 8 10 12 13 15 17 14 16 18 19 21 23 20 22 24],
                    2,
                    3,
                    4,
                )), 3, 2, 1)
        @finch begin
            x .= 0
            for i in _
                for j in _
                    for k in _
                        x[] += A[i, j, k]
                    end
                end
            end
        end

        @test x[] == (24 + 1) * 24 / 2

        @test check_output("index/swizzle_2.txt", @finch_code begin
            for i in _
                for j in _
                    for k in _
                        x[] += A[i, j, k]
                    end
                end
            end
        end)
    end
end

@testitem "masks" begin
    function mask_matrix(mask, m, n; rows=1:m)
        out = Tensor(Dense(SparseList(Element(false))), m, n)
        start = first(rows)
        stop = last(rows)
        @finch begin
            out .= false
            for j in 1:n, i in 1:m
                if start <= i <= stop
                    out[i, j] = mask[i, j]
                end
            end
        end
        Array(out)
    end

    function mask_vector(mask, n; rows=1:n)
        out = Tensor(SparseList(Element(false)), n)
        start = first(rows)
        stop = last(rows)
        @finch begin
            out .= false
            for i in 1:n
                if start <= i <= stop
                    out[i] = mask[i]
                end
            end
        end
        Array(out)
    end

    @testset "pair sums and carries" begin
        for n in (0, 1, 2, 7, 8)
            pairs = mask_matrix(pairsummask, cld(n, 2), n)
            carries = mask_matrix(paircarrymask, n, fld(n, 2))
            input = collect(1:n)
            @test pairs * input == [sum(input[i:min(i + 1, n)]) for i in 1:2:n]
            partial = collect(1:fld(n, 2))
            @test carries * partial == [i == 1 ? 0 : partial[fld(i, 2)] for i in 1:n]
        end
    end

    @testset "reverse, roll, and repeat" begin
        for n in (0, 1, 7), m in (0, 1, 5, 15)
            @test mask_matrix(reversemask(n), m, n) ==
                [j == n - i + 1 for i in 1:m, j in 1:n]
            for k in (-10, -1, 0, 1, 10)
                rolled = mask_matrix(rollmask(n, k), m, n)
                @test rolled == [n > 0 && j - 1 == mod(i - 1 - k, n) for i in 1:m, j in 1:n]
                if m == n
                    @test rolled * collect(1:n) == circshift(collect(1:n), k)
                end
            end
        end
        for k in (-1, 0, 1, 3), n in (0, 1, 5)
            repeated = mask_matrix(repeatmask(k), max(k, 0) * n, n)
            @test repeated * collect(1:n) == repeat(collect(1:n); inner=max(k, 0))
        end
        @test mask_matrix(repeatmask(0), 5, 3) == falses(5, 3)
        @test mask_matrix(repeatmask(-2), 5, 3) == falses(5, 3)
        @test mask_matrix(repeatmask(3), 7, 3) * [4, 5, 6] == [4, 4, 4, 5, 5, 5, 6]
        @test mask_matrix(rollmask(3, -1), 9, 3; rows=4:8) ==
            [4 <= i <= 8 && j == mod(i, 3) + 1 for i in 1:9, j in 1:3]
    end

    @testset "one-hot and parity" begin
        for n in (0, 1, 7), index in (-1, 0, 1, 3, 8)
            @test mask_vector(onehotmask(index), n) == [i == index for i in 1:n]
        end
        for n in (0, 1, 7), parity in (-1, 0, 1, 2)
            @test mask_vector(paritymask(parity), n) ==
                [mod(i - 1, 2) == parity for i in 1:n]
        end
        @test mask_vector(paritymask(), 9; rows=4:8) ==
            [false, false, false, false, true, false, true, false, false]
        @test mask_vector(paritymask(1), 9; rows=3:7) ==
            [false, false, false, true, false, true, false, false, false]
    end

    @testset "odd-even merge sort" begin
        for n in (0, 1, 2, 3, 7, 8, 13, 16)
            input = collect(n:-1:1)
            p = 1
            while p < n
                k = p
                while k >= 1
                    # Enumerate compare-exchanges as in the sorting network, using
                    # zero-based positions independently of the mask implementation.
                    partners = collect(1:n)
                    lower = falses(n)
                    for j in mod(k, p):(2k):(n - k - 1), i in 0:(k - 1)
                        a = i + j
                        b = a + k
                        if b < n && fld(a, 2p) == fld(b, 2p)
                            partners[a + 1] = b + 1
                            partners[b + 1] = a + 1
                            lower[a + 1] = true
                        end
                    end
                    mask = mask_matrix(oddevenmergesortpartnermask(n, p, k), n, n)
                    @test mask_vector(oddevenmergesortlowermask(n, p, k), n) == lower
                    @test mask == [j == partners[i] for i in 1:n, j in 1:n]
                    partner_values = mask * input
                    input =
                        ifelse.(
                            lower, min.(input, partner_values), max.(input, partner_values)
                        )
                    k = fld(k, 2)
                end
                p *= 2
            end
            @test input == collect(1:n)
        end
        @test mask_matrix(oddevenmergesortpartnermask(0, 1, 1), 0, 0) == falses(0, 0)
        @test mask_vector(oddevenmergesortlowermask(0, 1, 1), 0) == Bool[]
        @test_throws ArgumentError oddevenmergesortpartnermask(5, 0, 1)
        @test_throws ArgumentError oddevenmergesortlowermask(5, 1, 0)
    end

    @testset "reshape" begin
        for (old_shape, new_shape) in (
            ((2, 3), (3, 2)),
            ((6,), (2, 3)),
            ((2, 3), (6,)),
            ((1, 2, 3), (3, 2)),
            ((0, 3), (2, 0)),
            ((2, 0), (0,)),
            ((), ()),
            ((), (1, 1)),
            ((1, 1), ()),
        )
            mask = reshapemask(old_shape, new_shape)
            out = zeros(Bool, size(mask))
            copyto!(out, mask)
            # Reversing the dimensions makes Julia's column-major indexing
            # enumerate the source pattern's row-major positions.
            old_indices = LinearIndices(reverse(old_shape))
            new_indices = LinearIndices(reverse(new_shape))
            expected = zeros(Bool, size(mask))
            for old in CartesianIndices(old_shape), new in CartesianIndices(new_shape)
                expected[Tuple(old)..., Tuple(new)...] =
                    old_indices[reverse(Tuple(old))...] ==
                    new_indices[reverse(Tuple(new))...]
            end
            @test out == expected
            @test count(out) == prod(old_shape)
        end
        @test_throws DimensionMismatch reshapemask((2, 3), (5,))
        @test_throws ArgumentError reshapemask((-1,), (-1,))
    end
end

@testitem "randommask" begin
    using Random
    using StableRNGs

    rng = StableRNG(123)

    @testset "reproducible bit mixing" begin
        # Fixed outputs pin down the integer arithmetic and seed interpretation
        # independently of Julia's built-in hash and platform word size.
        @test Finch.randommask_mix(UInt64(0)) == 0xe220a8397b1dcdaf
        @test Finch.randommask_mix(UInt64(1)) == 0x910a2dec89025cc1
        @test Finch.randommask_mix(typemax(UInt64)) == 0xe4d971771b652c20
        @test Finch.randommask_uniform(UInt64(0)) == 0.0
        @test Finch.randommask_uniform(typemax(UInt64)) == prevfloat(1.0)
        mask = randommask(16, 0.5; seed=42)
        @test copyto!(zeros(Bool, 16), mask) ==
            Bool[0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        mask = randommask((4, 5), 0.5; seed=42)
        @test copyto!(zeros(Bool, 4, 5), mask) == Bool[
            1 0 1 0 0
            0 1 0 0 1
            0 1 0 1 0
            0 1 0 0 0
        ]
    end

    @testset "probabilities and shapes" begin
        for shape in ((), (0,), (3, 0), (1,), (4, 5), (2, 3, 4)), p in (0, 1)
            mask = randommask(rng, shape, p)
            out = fill(!Bool(p), shape)
            copyto!(out, mask)
            @test out == fill(Bool(p), shape)
            @test size(mask) == shape
            @test eltype(mask) == Bool
            @test Finch.fill_value(mask) == false
        end
        for shape in ((), (5,), (3, 4), (2, 3, 4))
            mask = randommask(rng, shape, 0.3)
            out = copyto!(zeros(Bool, shape), mask)
            for i in CartesianIndices(shape)
                @test out[i] == mask[Tuple(i)...]
            end
        end
        for p in (0.1, 0.25, 0.5, 0.9)
            sample = copyto!(zeros(Bool, 10000), randommask(10000, p; seed=42))
            @test abs(count(sample) / length(sample) - p) < 0.03
        end
        @test randommask(3, 0.5) isa Finch.RandomMask
    end

    @testset "stable reads and traversal" begin
        rng1 = StableRNG(456)
        mask = randommask(rng1, (7, 9), 0.4)
        same = randommask(StableRNG(456), (7, 9), 0.4)
        rng_after_construction = copy(rng1)
        @test mask.seed == same.seed
        @test mask.seed != randommask(StableRNG(457), (7, 9), 0.4).seed
        forward = copyto!(zeros(Bool, 7, 9), mask)
        @test forward == copyto!(zeros(Bool, 7, 9), same)
        @test forward == copyto!(zeros(Bool, 7, 9), randommask((7, 9), 0.4; seed=mask.seed))
        reversed_loops = zeros(Bool, 7, 9)
        @finch mode = :fast begin
            reversed_loops .= false
            for i in _, j in _
                reversed_loops[i, j] = mask[i, j]
            end
        end
        @test reversed_loops == forward
        for i in 7:-1:1, j in 9:-1:1
            @test mask[i, j] == forward[i, j]
        end
        sparse = copyto!(Tensor(Dense(SparseList(Element(false)))), mask)
        @test Array(sparse) == forward
        input = reshape(collect(1:63), 7, 9)
        total = Scalar(0)
        @finch begin
            total .= 0
            for j in _, i in _
                if mask[i, j]
                    total[] += input[i, j]
                end
            end
        end
        @test total() == sum(input[forward])
        randommask(rng1, 10, 0.5; seed=42)
        @test rand(rng1, UInt64) == rand(rng_after_construction, UInt64)
        @test occursin("seed=", summary(mask))
    end

    @testset "coordinate hashing" begin
        mask = randommask((64, 64), 0.5; seed=42)
        out = copyto!(zeros(Bool, 64, 64), mask)
        # Plain XOR of coordinates produces a symmetric matrix and a constant
        # diagonal. Mixing between coordinates must avoid those artifacts.
        @test out != transpose(out)
        @test 12 < sum(out[i, i] for i in 1:64) < 52
        @test out[:, 1] != out[:, 2]
        @test out[:, 1] != .!out[:, 2]
        larger = randommask((80, 90), 0.5; seed=42)
        grown = copyto!(zeros(Bool, 80, 90), larger)
        @test grown[1:64, 1:64] == out
        # No allocation or linearization of a shape whose product exceeds UInt64.
        large_shape = (typemax(Int), typemax(Int), 8)
        huge = randommask(large_shape, 0.5; seed=42)
        small = randommask((4, 5, 2), 0.5; seed=42)
        @test size(huge) == large_shape
        @test huge[3, 4, 2] == small[3, 4, 2]
        @test huge[large_shape...] == randommask(large_shape, 0.5; seed=42)[large_shape...]
        if Sys.WORD_SIZE == 64
            # In particular, indices above the old field-size limit are usable.
            large_index = Int(1) << 62
            @test huge[large_index, large_index, 8] isa Bool
        end
    end

    @testset "validation" begin
        for p in (-0.1, 1.1, NaN, Inf, -Inf)
            @test_throws ArgumentError randommask(rng, (2, 3), p)
        end
        @test_throws ArgumentError randommask(rng, (-1,), 0.5)
        for seed in (-1, 0.5, big(2)^64)
            @test_throws ArgumentError randommask(4, 0.5; seed=seed)
        end
        @test randommask(4, 0.5; seed=typemax(UInt64)).seed == typemax(UInt64)
    end
end
