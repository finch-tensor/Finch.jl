struct DefaultStyle end
struct UnknownStyle end

@nospecialize

result_style(a, b) = _result_style(a, b, combine_style(a, b), combine_style(b, a))
function _result_style(a, b, c::UnknownStyle, d::UnknownStyle)
    throw(MethodError(combine_style, (a, b)))
end
_result_style(a, b, c, d::UnknownStyle) = c
_result_style(a, b, c::UnknownStyle, d) = d
_result_style(a, b, c, d) = c #This is actually deterministic I think.
#_result_style(a, b, c::T, d::T) where {T} = (c == d) ? c : @assert false "TODO lower_style_ambiguity_error"
#_result_style(a, b, c, d) = (c == d) ? c : @assert false "TODO lower_style_ambiguity_error"
combine_style(a, b) = UnknownStyle()

combine_style(a::DefaultStyle, b) = b

@specialize
