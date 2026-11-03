@kwdef mutable struct Run
    body
end

Base.show(io::IO, ex::Run) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Run)
    print(io, "Run(body = ")
    print(io, ex.body)
    print(io, ")")
end

FinchNotation.finch_leaf(x::Run) = virtual(x)

struct RunStyle end

get_style(ctx, ::Run, root) = root.kind === loop ? RunStyle() : DefaultStyle()
combine_style(a::DefaultStyle, b::RunStyle) = RunStyle()
combine_style(a::LookupStyle, b::RunStyle) = RunStyle()
combine_style(a::ThunkStyle, b::RunStyle) = ThunkStyle()
combine_style(a::SimplifyStyle, b::RunStyle) = a
combine_style(a::RunStyle, b::RunStyle) = RunStyle()

function lower(ctx::AbstractCompiler, root::FinchNode, ::RunStyle)
    if root.kind === loop
        root = Rewrite(
            Postwalk(
                @rule access(~a::isvirtual, ~m, ~i..., ~j) => begin
                    a_2 = get_run_body(ctx, a.val, root.ext)
                    if a_2 != nothing
                        access(a_2, m, i...)
                    else
                        access(a, m, i..., j)
                    end
                end
            ),
        )(
            root
        )
        if get_style(ctx, root) isa RunStyle #TODO do we need this always? Can we do this generically?
            error("run style couldn't lower runs")
        end
        return ctx(root)
    else
        error("unimplemented")
    end
end

get_run_body(ctx, node, ext) = nothing
get_run_body(ctx, node::Run, ext) = node.body

#assume ssa

@kwdef mutable struct AcceptRun
    body
end

FinchNotation.finch_leaf(x::AcceptRun) = virtual(x)

Base.show(io::IO, ex::AcceptRun) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::AcceptRun)
    print(io, "AcceptRun(…)")
end

struct AcceptRunStyle end

get_style(ctx, ::AcceptRun, root) = root.kind === loop ? AcceptRunStyle() : DefaultStyle()
combine_style(a::DefaultStyle, b::AcceptRunStyle) = AcceptRunStyle()
combine_style(a::LookupStyle, b::AcceptRunStyle) = AcceptRunStyle()
combine_style(a::ThunkStyle, b::AcceptRunStyle) = ThunkStyle()
combine_style(a::SimplifyStyle, b::AcceptRunStyle) = a
combine_style(a::AcceptRunStyle, b::AcceptRunStyle) = AcceptRunStyle()
combine_style(a::RunStyle, b::AcceptRunStyle) = RunStyle()

function lower(ctx::AbstractCompiler, root::FinchNode, ::AcceptRunStyle)
    if root.kind === loop
        body = Rewrite(
            Postwalk(
                @rule access(~a::isvirtual, ~m, ~i..., ~j) => begin
                    a_2 = get_acceptrun_body(ctx, a.val, root.ext)
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
        if root.idx in PostOrderDFS(body)
            #The loop body isn't constant after removing AcceptRuns, lower with a for-loop
            return ctx(root, LookupStyle())
        else
            #The loop body is constant after removing AcceptRuns, lower only the body once
            return ctx(body)
        end
    elseif root.kind === block
        quote end #TODO this shouldn't need to be specified
    else
        error("unimplemented")
    end
end

get_acceptrun_body(ctx, node, ext) = nothing
get_acceptrun_body(ctx, node::AcceptRun, ext) = node.body(ctx, ext)

function get_point_body(ctx, node::AcceptRun, ext, idx)
    node.body(ctx, similar_extent(ext, idx, idx))
end
