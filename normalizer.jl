using Finch


function find_normalizer_split(lvl::SparseListLevel, active, tid, dim, bound, target_work)
    max_idx = lvl.shape
    lo = 1
    hi = max_idx
    pos = dim > 1 ? bound[dim - 1] : 1

    while lo <= hi
        candidate = div(lo + hi, 2)
        stored = 0 ##Will countstored on (pos, mid)
        if stored >=
        end
    end
end