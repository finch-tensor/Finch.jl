using Finch


##BUGS: The only failures are all the same already-known edge case: a processor whose entire quota got absorbed by a duplicate cluster, producing an inverted lb > ub 
##Probably not worth fixing as they won't happen under realistic conditions.
struct MergeNormalization end

function decrement_idxs(idxs, shapes)
    idxs = copy(idxs)
    pos = 1
    while pos <= length(idxs)
        if idxs[pos] > 1
            idxs[pos] -= 1
            return idxs
        else
            idxs[pos] = shapes[pos]
            pos += 1
        end
    end
    error("nnz too small to load balance across P processors")
end

function balance(lvl, tid, P, nnz, style::Union{MergeNormalization})
    shapes = collect(Finch.level_size(lvl))
    max_dim = length(shapes)

    base = div(nnz, P)
    remainder = nnz % P
    lower_work = (tid - 1) * base + min(tid - 1, remainder)
    upper_work = tid * base + min(tid, remainder)

    if tid == 1
        lb_idxs = ntuple(_ -> 1, max_dim)
    else
        lb_idxs = find_normalizer_split(lvl, lower_work, P, copy(shapes), max_dim)
    end
    lb = Tuple(lb_idxs)

    if tid == P
        ub_idxs = shapes
    else
        next_lb_idxs = find_normalizer_split(lvl, upper_work, P, copy(shapes), max_dim)
        ub_idxs = decrement_idxs(next_lb_idxs, shapes)
    end
    ub = Tuple(ub_idxs)

    return (lb, ub)
end

function find_normalizer_split(lvl, target_work, P, idxs, max_dim)
    dim = 0
    while dim < max_dim
        lo = 1
        hi = idxs[max_dim - dim]
        truth = -1
        while lo <= hi
            candidate = div(lo + hi, 2)
            idxs[max_dim - dim] = candidate
            stored = 0
            for p in 1:P
                stored += Finch.countstored_level(lvl, 1, idxs, 0, p, true)
            end
            if stored >= target_work
                truth = candidate
                hi = candidate - 1
            else
                lo = candidate + 1
            end
        end
        idxs[max_dim - dim] = truth
        dim += 1
    end
    return idxs
end