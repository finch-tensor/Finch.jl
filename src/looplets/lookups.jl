#Note that lookups are lowered by the DefaultStyle loop lowerer in lower.jl

@kwdef struct Lookup
    body
end

Base.show(io::IO, ex::Lookup) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Lookup)
    print(io, "Lookup()")
end

FinchNotation.finch_leaf(x::Lookup) = virtual(x)

struct LookupStyle end

get_style(ctx, ::Lookup, root) = root.kind === loop ? LookupStyle() : DefaultStyle()

combine_style(a::DefaultStyle, b::LookupStyle) = LookupStyle()
combine_style(a::ThunkStyle, b::LookupStyle) = ThunkStyle()
combine_style(a::SimplifyStyle, b::LookupStyle) = a
combine_style(a::LookupStyle, b::LookupStyle) = LookupStyle()

function lower(ctx::AbstractCompiler, root::FinchNode, ::LookupStyle)
    if root.kind === loop
        idx_sym = freshen(ctx, root.idx.name)
        body = contain(ctx) do ctx_2
            set_binding!(ctx_2, root.idx, value(idx_sym))
            body_3 = Rewrite(
                Postwalk(
                    @rule access(~a::isvirtual, ~m, ~i..., ~j) => begin
                        a_2 = get_point_body(ctx_2, a.val, root.ext.val, value(idx_sym))
                        if a_2 != nothing
                            access(a_2, m, i...)
                        else
                            access(a, m, i..., j)
                        end
                    end
                ),
            )(
                root.body
            )
            open_scope(ctx_2) do ctx_3
                ctx_3(body_3)
            end
        end
        @assert isvirtual(root.ext)

        target = is_continuous_extent(root.ext) ? 0 : 1
        if prove(ctx, call(==, measure(root.ext.val), target))
            return quote
                $idx_sym = $(ctx(getstart(root.ext)))
                $body
            end
        else
            return quote
                for $idx_sym in ($(ctx(getstart(root.ext)))):($(ctx(getstop(root.ext))))
                    $body
                end
            end
        end
    else
        error("unimplemented")
    end
end

get_point_body(ctx, node::Lookup, ext, idx) = node.body(ctx, idx)

get_point_body(ctx, node, ext, idx) = nothing
