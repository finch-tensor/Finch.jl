using AcceleratedKernels

Base.@propagate_inbounds function s_prefix_sum(arr::Vector{Int})
    out = Vector{Int}(undef, length(arr))
    out[1] = arr[1]
    for i in 2:length(arr)
        out[i] = out[i - 1] + arr[i]
    end
    out
end

Base.@propagate_inbounds function binary_search(target::Int, arr)
    lo = 1
    hi = length(arr)
    @assert target > 0
    @assert target < arr[hi]

    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] <= target && arr[mid + 1] > target
            return mid
        elseif arr[mid] > target
            hi = mid
        else
            lo = mid
        end
    end

    return -1
end

Base.@propagate_inbounds function binary_search_scalar(target, arr)
    lo = 1
    hi = length(arr)

    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] == target
            return mid
        elseif arr[mid] > target
            hi = mid - 1
        else
            lo = mid + 1
        end
    end

    return -1
end

Base.@propagate_inbounds function compute_proc_cutoffs(index, P)
    cutoffs = Vector{Int}(undef, length(index) + 1)
    cutoffs[1] = 1
    Threads.@threads for i in 2:length(cutoffs)
        cutoffs[i] = length(index[i-1])
    end
    s_prefix_sum(cutoffs)
end

Base.@propagate_inbounds function get_permute_idx(proc_id, ptr)
    start = 0

    for i in 1:proc_id - 1
        start += length(ptr[i]) - 1
    end

    return start
end

Base.@propagate_inbounds function p_permute(permutation, arr::Vector{T}) where {T}
    shuffled = Vector{T}(undef, length(arr))

    @assert length(permutation) == length(arr)

    Threads.@threads for i in eachindex(permutation)
        shuffled[i] = arr[permutation[i]]
    end

    return shuffled
end

#### Dense Merging Functions

Base.@propagate_inbounds function merge_dense(global_fbr_map, local_fbr_map, task_map, factor, shape, P)
    return global_fbr_map, local_fbr_map, task_map, factor * shape
end

Base.@propagate_inbounds function unroll_dense_coalesce(global_fbr_map, local_fbr_map, task_map, factor, P)
    unrolled_size = factor * length(global_fbr_map)

    global_fbr_map_unrolled = Vector{Int64}(undef, unrolled_size)
    local_fbr_map_unrolled = Vector{Int64}(undef, unrolled_size)
    task_map_unrolled = Vector{Int64}(undef, unrolled_size)

    chk_size = fld(length(global_fbr_map) + P - 1, P)

    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        for i in 0:chk_size - 1
            offset = init + i

            if offset > length(global_fbr_map)
                break
            end
            start_write = (offset - 1) * factor + 1
            finish_write = offset * factor

            gfbr = (global_fbr_map[offset] - 1) * factor + 1
            lfbr = (local_fbr_map[offset] - 1) * factor + 1
            task = task_map[offset]

            for j in start_write:finish_write
                global_fbr_map_unrolled[j] = gfbr
                local_fbr_map_unrolled[j] = lfbr
                task_map_unrolled[j] = task

                gfbr += 1
                lfbr += 1
            end
        end
    end

    return global_fbr_map_unrolled, local_fbr_map_unrolled, task_map_unrolled
end

#### SparseList Merging Functions

Base.@propagate_inbounds function gen_pos_idx_map(global_fbr_map, local_fbr_map, task_map, ptr, index, cutoffs, P)
    ordering = Base.Order.By(j -> (task_map[j], local_fbr_map[j]))
    sorter = AcceleratedKernels.sortperm(collect(1:length(task_map)); order=ordering)

    nnz = cutoffs[length(cutoffs)] - 1
    merged_positions = Vector{Int}(undef, nnz)
    merged_indices = Vector{Int}(undef, nnz)

    task_map2 = Vector{Int}(undef, nnz)
    local_fbr_map2 = Vector{Int}(undef, nnz)

    chk_size = fld(nnz + P - 1, P)
    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        proc_id = binary_search(init, cutoffs)
        idx_id = init - cutoffs[proc_id] + 1

        local_fbr = binary_search(idx_id, ptr[proc_id])

        tag = get_permute_idx(proc_id, ptr) + local_fbr

        @assert local_fbr > 0
        @assert tag > 0

        global_fbr = global_fbr_map[sorter[tag]]
        local_fbr_id_child = init - cutoffs[proc_id] + 1
        
        j = 0
        for i in 0:chk_size - 1
            offset = init + i
            if offset > nnz
                break
            end
            
            nz_id = j + idx_id
            idx = index[proc_id][nz_id]
            merged_positions[offset] = global_fbr
            merged_indices[offset] = idx
            task_map2[offset] = proc_id
            local_fbr_map2[offset] = local_fbr_id_child

            if nz_id >= length(index[proc_id]) && proc_id < P
                proc_id += 1
                idx_id = 1
                j = 0
                local_fbr_id_child = 1
                local_fbr = 1

                tag += 1
                global_fbr = global_fbr_map[sorter[tag]]
            elseif nz_id + 1 >= ptr[proc_id][local_fbr + 1] && local_fbr + 1 < length(ptr[proc_id])
                local_fbr += 1

                tag += 1
                global_fbr = global_fbr_map[sorter[tag]]
                local_fbr_id_child += 1
                j += 1
            else
                j += 1
                local_fbr_id_child += 1
            end
        end
    end
    return merged_positions, merged_indices, local_fbr_map2, task_map2
end

Base.@propagate_inbounds function process_next_lvl(merged_positions, merged_indices, task_map, local_fbr_map, P)
    ordering = Base.Order.By(j -> (merged_positions[j], merged_indices[j]))
    shuffler = AcceleratedKernels.sortperm(collect(1:length(merged_positions)); order=ordering)

    nnz = length(local_fbr_map)
    global_fbr_map2 = Vector{Int}(undef, nnz)

    merged_positions_s = p_permute(shuffler, merged_positions)
    merged_indices_s = p_permute(shuffler, merged_indices)
    task_map = p_permute(shuffler, task_map)
    local_fbr_map = p_permute(shuffler, local_fbr_map)

    uq_ptr = zeros(Int, P + 1)
    uq_idx = zeros(Int, P + 1)

    chk_size = fld(nnz + P - 1, P)

    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        seen = 0
        prev = init > 1 ? (merged_positions_s[init-1], merged_indices_s[init-1]) : (-1, -1)
        prev_ptr = init > 1 ? merged_positions_s[init-1] : 1
        seen_ptr = 0

        for i in 0:chk_size-1
            offset = init + i
            if offset > nnz
                break
            end

            tup = (merged_positions_s[offset], merged_indices_s[offset])
            if tup != prev
                prev = tup
                seen += 1
            end

            p = merged_positions_s[offset]
            if prev_ptr != p
                seen_ptr += (p - prev_ptr)
                prev_ptr = p
            end
        end
        uq_idx[tid+1] = seen
        uq_ptr[tid+1] = seen_ptr
    end
    uq_ptr_s = s_prefix_sum(uq_ptr)
    uq_idx_s = s_prefix_sum(uq_idx)

    lvl_ptr = zeros(Int, merged_positions_s[length(merged_positions_s)] + 1)
    lvl_idx = zeros(Int, uq_idx_s[length(uq_idx_s)])

    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        seen_ptr = uq_ptr_s[tid] + 2
        seen_idx = uq_idx_s[tid] + 1
        prev = init > 1 ? (merged_positions_s[init-1], merged_indices_s[init-1]) : (1, -1)

        for i in 0:chk_size-1
            offset = init + i
            if offset > nnz
                break
            end

            while seen_ptr < merged_positions_s[offset]
                lvl_ptr[seen_ptr] = seen_idx
                seen_ptr += 1
            end

            tup = (merged_positions_s[offset], merged_indices_s[offset])
            if tup != prev
                lvl_idx[seen_idx] = tup[2]

                p = merged_positions_s[offset]
                if prev[1] != p
                    lvl_ptr[seen_ptr] = seen_idx
                    seen_ptr += 1
                end
                prev = tup
                seen_idx += 1
            end
            global_fbr_map2[offset] = seen_idx - 1
        end

        if tid == P
            lvl_ptr[length(lvl_ptr)] = length(lvl_idx) + 1
            lvl_ptr[1] = 1
        end
    end

    return global_fbr_map2, local_fbr_map, task_map, lvl_ptr, lvl_idx
end

#### ElementLevel Merge Functions

Base.@propagate_inbounds function merge_element_level(global_fbr_map, local_fbr_map, task_map, val, P)
    chk_size = fld(length(global_fbr_map) + P - 1, P)
    val2 = zeros(global_fbr_map[length(global_fbr_map)])

    Threads.@threads for tid in 1:P
        start, finish = 0, 0

        if tid > 1
            offset_start = (tid - 1) * chk_size + 1
            last_idx = global_fbr_map[offset_start - 1]

            while offset_start > 1 && global_fbr_map[offset_start] == last_idx
                offset_start += 1
            end
            start = offset_start
        else
            start = 1
        end

        if tid < P
            offset_finish = tid * chk_size + 1
            last_idx = global_fbr_map[offset_finish - 1]

            while offset_finish <= length(global_fbr_map) && global_fbr_map[offset_finish] == last_idx
                offset_finish += 1
            end
            finish = offset_finish
        else
            finish = length(global_fbr_map) + 1
        end

        for i in start:finish - 1
            @fastmath val2[global_fbr_map[i]] += val[task_map[i]][local_fbr_map[i]]
        end
    end

    return val2
end

Base.@propagate_inbounds function merge_dense_element_level(global_fbr_map, local_fbr_map, task_map, factor, val, P)
    val2 = zeros(global_fbr_map[length(global_fbr_map)] * factor)
    chk_size = fld(factor + P - 1, P)
    iter_range = length(global_fbr_map)

    Threads.@threads for tid in 1:P
        init_dense_fbr = (tid - 1) * chk_size
        if init_dense_fbr >= factor
            break
        end
        for i in 1:iter_range
            val2_offset = (global_fbr_map[i] - 1) * factor + 1
            val2_local_offset = (local_fbr_map[i] - 1) * factor + 1
            for dense_fbr in init_dense_fbr:init_dense_fbr + chk_size - 1
                @fastmath val2[val2_offset + dense_fbr] += val[task_map[i]][val2_local_offset + dense_fbr]
            end
        end
    end

    return val2
end