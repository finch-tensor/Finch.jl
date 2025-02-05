let
    formats = []
    Ts = [Int, Float64]#, Bool]

    tik = time()
    for (n, T) in enumerate(Ts)
        if n > 1
            tok = time()
            estimated = ceil(Int, (tok - tik) / (n - 1) * (length(Ts) - n + 1))
            @info "Precompiling common tensor formats... (estimated: $(fld(estimated, 60)) minutes and $(mod(estimated, 60)) seconds)"
        else
            @info "Precompiling common tensor formats..."
        end
        f = zero(T)
        append!(
            formats,
            [
                Scalar(f, rand(T)),
                Tensor(Dense(Element(f)), rand(T, 2)),
                Tensor(SparseList(Element(f)), rand(T, 2)),
                Tensor(Dense(Dense(Element(f))), rand(T, 2, 2)),
                Tensor(Dense(SparseList(Element(f))), rand(T, 2, 2)),
            ],
        )
    end

    for (n, format) in enumerate(formats)
        if n > 1
            tok = time()
            estimated = ceil(Int, (tok - tik) / (n - 1) * (length(formats) - n + 1))
            @info "Precompiling common tensor operations... (estimated: $(fld(estimated, 60)) minutes and $(mod(estimated, 60)) seconds)"
        else
            @info "Precompiling common tensor operations..."
        end
        A = deepcopy(format)
        B = deepcopy(format)

        if ndims(format) > 0
            dropfills(A)
            copyto!(A, B)
        end
        A == B
        i = rand(1:2, ndims(A))
        A[i...]
        #if eltype(format) == Bool
        #    .!(A)
        #    any(A)
        #    all(A)
        #end
        if eltype(format) <: Integer
            .~(A)
            A .& B
            A .| B
        end
        if eltype(format) <: Union{Integer,AbstractFloat} && eltype(format) != Bool
            sum(A)
            A .* B
            A + A
            A - A
            maximum(A)
            max.(A, B)
            println("")
            for T in Ts
                A * rand(T)
                A + rand(T)
            end
            if ndims(A) == 2
                A * A
                A * Tensor(Dense(Element(zero(eltype(A)))), rand(eltype(A), 2))
            end
        end
    end

    @info "Done!"
end
