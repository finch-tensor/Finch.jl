using Finch
using Random

# ---------------------------------------------------------------------------
# Specialized tests for TupleMask, targeting the change that removed the
# runtime `reverse(lb)`/`reverse(ub)` from virtualize() in favor of walking
# dimensions from N down to 1 directly (indexing the ORIGINAL, unreversed
# tuple). These tests check:
#   1. Semantics are unchanged vs. the documented behavior (docstring
#      example, and general lexicographic brute force, for 2D and 3D).
#   2. Edge cases: single-point ranges, ties across multiple leading dims,
#      full-range bounds.
#   3. The generated code no longer contains a `reverse` call.
# ---------------------------------------------------------------------------

# lexicographic comparison matching the docstring: "most significant digits
# start on the rightmost end of the tuple" -- i.e. compare last-to-first.
function lex_between(lb, ub, idx)
    return reverse(lb) <= reverse(idx) <= reverse(ub)
end

function materialize_mask(lb, ub, shape)
    N = length(shape)
    m = Finch.tuplemask(lb, ub)
    if N == 2
        S1, S2 = shape
        M = Tensor(Dense(Dense(Element(false))), S1, S2)
        @finch begin
            M .= false
            for j in _
                for i in _
                    M[i, j] = m[i, j]
                end
            end
        end
    elseif N == 3
        S1, S2, S3 = shape
        M = Tensor(Dense(Dense(Dense(Element(false)))), S1, S2, S3)
        @finch begin
            M .= false
            for k in _
                for j in _
                    for i in _
                        M[i, j, k] = m[i, j, k]
                    end
                end
            end
        end
    else
        error("unsupported ndims for this test: $N")
    end
    return Array(M)
end

issues = String[]

# ---------------------------------------------------------------------------
# 1. Docstring semantics: tuplemask compares lexicographically with the LAST
#    tuple entry most significant, e.g. (4,1,6) < (1,1,7) because 6 < 7.
# ---------------------------------------------------------------------------
let
    lb = (2, 1, 6)
    ub = (5, 3, 6)
    # (4,1,6) should be inside [lb,ub]? reverse-compare: (6,1,4) between (6,1,2) and (6,3,5)? let's just use lex_between directly against a few points
    for (idx, want) in [
        ((4, 1, 6), true),   # matches ub's outer/most-significant dim (6), within (1,3) mid, within (2,5) inner? check via lex_between
        ((1, 1, 7), false),  # outer dim 7 > 6, out of range entirely
        ((2, 1, 6), true),   # exactly lb
        ((5, 3, 6), true),   # exactly ub
        ((1, 1, 6), false),  # inner dim below lb's inner bound while mid/outer tied to lb
        ((6, 3, 6), false),  # inner dim above ub's inner bound while mid/outer tied to ub
    ]
        got = materialize_mask(lb, ub, (6, 3, 7))[idx...]
        want2 = lex_between(lb, ub, idx)
        if want2 != want
            push!(issues, "docstring-sanity: my own `want` disagrees with lex_between for $idx (want=$want, lex_between=$want2) -- check test authoring")
        end
        if got != want2
            push!(issues, "docstring semantics: tuplemask$(idx) = $got, expected $want2 for lb=$lb ub=$ub")
        end
    end
end

# ---------------------------------------------------------------------------
# 2. Exhaustive brute-force check over small full grids, 2D and 3D, many
#    random (lb,ub) pairs (lb <= ub lexicographically by construction).
# ---------------------------------------------------------------------------
function random_lex_bounds(shape, rng)
    N = length(shape)
    a = ntuple(d -> rand(rng, 1:shape[d]), N)
    b = ntuple(d -> rand(rng, 1:shape[d]), N)
    return reverse(a) <= reverse(b) ? (a, b) : (b, a)
end

rng = MersenneTwister(42)
for shape in [(5, 4), (3, 3), (7, 2), (4, 5, 3), (2, 2, 2), (6, 3, 2)]
    for trial in 1:8
        lb, ub = random_lex_bounds(shape, rng)
        got = materialize_mask(lb, ub, shape)
        N = length(shape)
        for idx in Iterators.product((1:s for s in shape)...)
            want = lex_between(lb, ub, idx)
            if got[idx...] != want
                push!(issues, "brute-force shape=$shape lb=$lb ub=$ub idx=$idx: got=$(got[idx...]) want=$want")
            end
        end
    end
end

# ---------------------------------------------------------------------------
# 3. Edge cases: single point, full range, ties spanning all-but-one dim.
# ---------------------------------------------------------------------------
let
    shape = (4, 4, 4)
    cases = [
        (shape, shape),                    # single point at the very last coord
        ((1, 1, 1), (1, 1, 1)),            # single point at the very first coord
        ((1, 1, 1), shape),                # full range
        ((2, 3, 2), (2, 3, 2)),            # single interior point
        ((1, 1, 2), (4, 4, 2)),            # tie on outermost dim only
        ((3, 1, 2), (3, 4, 4)),            # tie on outermost+middle dims
    ]
    for (lb, ub) in cases
        got = materialize_mask(lb, ub, shape)
        for idx in Iterators.product((1:s for s in shape)...)
            want = lex_between(lb, ub, idx)
            if got[idx...] != want
                push!(issues, "edge-case shape=$shape lb=$lb ub=$ub idx=$idx: got=$(got[idx...]) want=$want")
            end
        end
    end
end

# ---------------------------------------------------------------------------
# 4. Structural check: the compiled code must no longer contain `reverse`.
# ---------------------------------------------------------------------------
let
    m = Finch.tuplemask((1, 1), (3, 3))
    M = Tensor(Dense(Dense(Element(false))), 4, 4)
    code = string(@finch_code begin
        M .= false
        for j in _
            for i in _
                M[i, j] = m[i, j]
            end
        end
    end)
    if occursin("reverse", code)
        push!(issues, "generated code for tuplemask still contains a `reverse` call:\n$code")
    else
        println("structural check: no `reverse` in generated code -- OK")
    end
end

println()
if isempty(issues)
    println("ALL SPECIALIZED TUPLEMASK TESTS PASSED")
else
    println("ISSUES FOUND ($(length(issues))):")
    for iss in issues
        println("  $iss")
    end
end
