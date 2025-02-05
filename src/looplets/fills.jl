struct FillLeaf
    body::FinchNode
    FillLeaf(x) = new(finch_leaf(x))
end

FinchNotation.finch_leaf(x::FillLeaf) = virtual(x)
virtual_fill_value(ctx, f::FillLeaf) = f.body

struct FillStyle end

get_style(ctx, ::FillLeaf, root) = FillStyle()

combine_style(a::DefaultStyle, b::FillStyle) = FillStyle()
combine_style(a::LookupStyle, b::FillStyle) = FillStyle()
combine_style(a::ThunkStyle, b::FillStyle) = FillStyle()
combine_style(a::SimplifyStyle, b::FillStyle) = a
combine_style(a::RunStyle, b::FillStyle) = FillStyle()
combine_style(a::AcceptRunStyle, b::FillStyle) = FillStyle()
combine_style(a::SpikeStyle, b::FillStyle) = FillStyle()
combine_style(a::FillStyle, b::SimplifyStyle) = FillStyle()
combine_style(a::FillStyle, b::FillStyle) = FillStyle()
combine_style(a::FillStyle, b::StepperStyle) = FillStyle()
combine_style(a::FillStyle, b::JumperStyle) = FillStyle()

function lower(ctx::AbstractCompiler, root::FinchNode, ::FillStyle)
    ctx(
        Postwalk(
            @rule access(~a::isvirtual, ~m, ~i...) =>
                visit_fill_leaf_leaf(access(a, m, i...), a.val)
        )(
            root
        ),
    )
end

visit_fill_leaf_leaf(node, tns) = nothing
visit_fill_leaf_leaf(node, tns::FillLeaf) = Simplify(tns.body)
