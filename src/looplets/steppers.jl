@kwdef struct StepperStyle
    count = 1
end

@kwdef struct Stepper
    preamble = nothing
    stop = (ctx, ext) -> nothing
    chunk = nothing
    next = (ctx, ext) -> nothing
    body = (ctx, ext) -> chunk
    seek = (ctx, start) -> error("seek not implemented error")
end

Base.show(io::IO, ex::Stepper) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Stepper)
    print(io, "Stepper()")
end

FinchNotation.finch_leaf(x::Stepper) = virtual(x)

get_style(ctx, ::Stepper, root) = root.kind === loop ? StepperStyle() : DefaultStyle()
combine_style(a::DefaultStyle, b::StepperStyle) = b
combine_style(a::LookupStyle, b::StepperStyle) = b
combine_style(a::StepperStyle, b::SequenceStyle) = SequenceStyle()
combine_style(a::StepperStyle, b::StepperStyle) = StepperStyle(a.count + b.count)
combine_style(a::StepperStyle, b::RunStyle) = RunStyle()
combine_style(a::SimplifyStyle, b::StepperStyle) = a
combine_style(a::StepperStyle, b::AcceptRunStyle) = a
combine_style(a::StepperStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::StepperStyle, b::SwitchStyle) = SwitchStyle()
combine_style(a::ThunkStyle, b::StepperStyle) = ThunkStyle()
combine_style(a::StepperStyle, b::JumperStyle) = JumperStyle()
combine_style(a::StepperStyle, b::PhaseStyle) = b

stepper_seek(ctx, node::Stepper, ext) = node.seek(ctx, ext)
stepper_seek(ctx, node, ext) = quote end

stepper_range(ctx, node, ext) = auto

function stepper_range(ctx, node::FinchNode, ext)
    if @capture node access(~tns::isvirtual, ~i...)
        stepper_range(ctx, tns.val, ext)
    else
        return auto
    end
end

function stepper_range(ctx, node::Stepper, ext)
    push_preamble!(ctx, node.preamble !== nothing ? node.preamble : quote end)
    ext_2 = similar_extent(ext, getstart(ext), node.stop(ctx, ext))
    bound_measure_below!(ext_2, get_smallest_measure(ext))
end

stepper_body(ctx, node, ext, ext_2) = truncate(ctx, node, ext, ext_2)

function stepper_body(ctx, node::FinchNode, ext, ext_2)
    if @capture node access(~tns::isvirtual, ~m, ~i...)
        access(stepper_body(ctx, tns.val, ext, ext_2), m, i...)
    else
        return node
    end
end

function stepper_body(ctx, node::Stepper, ext, ext_2)
    next = node.next(ctx, ext_2)
    if next !== nothing
        full_chunk = Thunk(;
            body=(ctx) -> truncate(
                ctx, node.chunk, ext, similar_extent(ext, getstart(ext_2), getstop(ext))
            ),
            epilogue=next,
        )
        truncated_chunk = truncate(
            ctx,
            node.chunk,
            ext,
            similar_extent(
                ext,
                getstart(ext_2),
                bound_above!(getstop(ext_2), call(-, getstop(ext), getunit(ext))),
            ),
        )
        if prove(ctx, call(<=, node.stop(ctx, ext), getstop(ext_2)))
            full_chunk
        elseif prove(ctx, call(>=, node.stop(ctx, ext), getstop(ext_2)))
            truncated_chunk
        else
            Switch([
                value(:($(ctx(node.stop(ctx, ext))) == $(ctx(getstop(ext_2))))) =>
                    full_chunk,
                literal(true) => truncated_chunk,
            ])
        end
    else
        node.body(ctx, ext_2)
    end
end

function lower(ctx::AbstractCompiler, root::FinchNode, style::StepperStyle)
    root.kind === loop || error("unimplemented")

    i = getname(root.idx)
    i0 = freshen(ctx, i, :_start)
    push_preamble!(
        ctx,
        quote
            $i = $(ctx(getstart(root.ext)))
        end,
    )

    guard = :($i <= $(ctx(getstop(root.ext))))

    foreach(filter(isvirtual, collect(PostOrderDFS(root.body)))) do node
        push_preamble!(ctx, stepper_seek(ctx, node.val, root.ext))
    end

    if style.count == 1 &&
        !prove(ctx, call(==, measure(root.ext.val), get_smallest_measure(root.ext.val)))
        body_2 = contain(ctx) do ctx_2
            push_preamble!(ctx_2, :($i0 = $i))
            i1 = freshen(ctx_2, i)

            ext_1 = bound_measure_below!(
                similar_extent(root.ext, value(i0), getstop(root.ext)),
                get_smallest_measure(root.ext),
            )
            ext_2 = mapreduce(
                (node) -> stepper_range(ctx_2, node, ext_1),
                (a, b) -> virtual_intersect(ctx_2, a, b),
                PostOrderDFS(root.body),
            )
            ext_3 = virtual_intersect(ctx_2, ext_1, ext_2)
            ext_5 = cache_dim!(ctx_2, :phase, ext_2)

            full_body = Rewrite(Postwalk(node -> stepper_body(ctx_2, node, ext_1, ext_2)))(
                root.body
            )
            full_body = quote
                $i1 = $i
                $(
                    contain(ctx_2) do ctx_3
                        ctx_3(loop(root.idx, ext_5, full_body))
                    end
                )

                $i = $(ctx_2(getstop(ext_5))) + $(ctx_2(getunit(ext_5)))
            end

            truncated_body = contain(ctx_2) do ctx_3
                ext_4 = cache_dim!(ctx_3, :phase, ext_3)
                truncated_body = Rewrite(
                    Postwalk(node -> stepper_body(ctx_3, node, ext_1, ext_4))
                )(
                    root.body
                )
                truncated_body = quote
                    $i1 = $i
                    $(
                        contain(ctx_3) do ctx_4
                            ctx_4(loop(root.idx, ext_4, truncated_body))
                        end
                    )

                    $i = $(ctx_3(getstop(ext_4))) + $(ctx_3(getunit(ext_4)))
                end

                truncated_body = if prove(ctx_3, call(>=, measure(ext_4), 0))
                    truncated_body
                else
                    quote
                        if $(ctx_3(getstop(ext_4))) >= $(ctx_3(getstart(ext_4)))
                            $truncated_body
                        end
                    end
                end
            end

            quote
                if $(ctx_2(getstop(ext_5))) < $(ctx(getstop(root.ext)))
                    $full_body
                else
                    $truncated_body
                    break
                end
            end
        end

        @assert isvirtual(root.ext)

        cases = quote end
        for (guard, root_2) in ShortCircuitVisitor(ctx)(root)
            push!(
                cases.args,
                quote
                    if $guard
                        $(
                            contain(ctx) do ctx_2
                                ext_1 = bound_measure_below!(
                                    similar_extent(root.ext, value(i), getstop(root.ext)),
                                    get_smallest_measure(root.ext),
                                )
                                ctx_2(loop(root.idx, ext_1, root_2))
                            end
                        )
                        break
                    end
                end,
            )
        end

        if prove(ctx, call(==, measure(root.ext.val), get_smallest_measure(root.ext.val)))
            body_2
        else
            return quote
                while true
                    $cases
                    $body_2
                end
            end
        end

    else
        body_2 = contain(ctx) do ctx_2
            push_preamble!(ctx_2, :($i0 = $i))
            i1 = freshen(ctx_2, i)

            ext_1 = bound_measure_below!(
                similar_extent(root.ext, value(i0), getstop(root.ext)),
                get_smallest_measure(root.ext),
            )
            ext_2 = mapreduce(
                (node) -> stepper_range(ctx_2, node, ext_1),
                (a, b) -> virtual_intersect(ctx_2, a, b),
                PostOrderDFS(root.body),
            )
            ext_3 = virtual_intersect(ctx_2, ext_1, ext_2)
            ext_4 = cache_dim!(ctx_2, :phase, ext_3)

            body = Rewrite(Postwalk(node -> stepper_body(ctx_2, node, ext_1, ext_4)))(
                root.body
            )
            body = quote
                $i1 = $i
                $(
                    contain(ctx_2) do ctx_3
                        ctx_3(loop(root.idx, ext_4, body))
                    end
                )

                $i = $(ctx_2(getstop(ext_4))) + $(ctx_2(getunit(ext_4)))
            end

            if prove(ctx_2, call(>=, measure(ext_4), 0))
                body
            else
                quote
                    if $(ctx_2(getstop(ext_4))) >= $(ctx_2(getstart(ext_4)))
                        $body
                    end
                end
            end
        end

        @assert isvirtual(root.ext)

        cases = quote end
        for (guard, root_2) in ShortCircuitVisitor(ctx)(root)
            push!(
                cases.args,
                quote
                    if $guard
                        $(
                            contain(ctx) do ctx_2
                                ext_1 = bound_measure_below!(
                                    similar_extent(root.ext, value(i), getstop(root.ext)),
                                    get_smallest_measure(root.ext),
                                )
                                ctx_2(loop(root.idx, ext_1, root_2))
                            end
                        )
                        break
                    end
                end,
            )
        end

        if prove(ctx, call(==, measure(root.ext.val), get_smallest_measure(root.ext.val)))
            body_2
        else
            return quote
                while $guard
                    $cases
                    $body_2
                end
            end
        end
    end
end
