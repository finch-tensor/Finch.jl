struct Permit end

Base.show(io::IO, ex::Permit) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Permit)
	print(io, "Permit()")
end

const permit = Permit()

Base.getindex(arr::Permit, i) = i

struct VirtualPermit end

Base.show(io::IO, ex::VirtualPermit) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualPermit)
	print(io, "VirtualPermit()")
end

FinchNotation.finch_leaf(x::VirtualPermit) = virtual(x)

virtualize(ex, ::Type{Permit}, ctx) = VirtualPermit()

(ctx::LowerJulia)(tns::VirtualPermit) = :(Permit())

virtual_size(arr::VirtualPermit, ctx::LowerJulia, dim) = (widendim(dim),)
virtual_resize!(arr::VirtualPermit, ctx::LowerJulia, idx_dim) = widendim(idx_dim)
virtual_eldim(arr::VirtualPermit, ctx::LowerJulia, idx_dim) = widendim(idx_dim)

function get_reader(::VirtualPermit, ctx, proto_idx)
    Furlable(
        size = (nodim,),
        body = nothing,
        fuse = (tns, ctx, ext) -> Pipeline([
            Phase(
                stride = (ctx, ext_2) -> call(-, getstart(ext), 1),
                body = (ctx, ext) -> Run(Fill(literal(missing))),
            ),
            Phase(
                stride = (ctx, ext_2) -> getstop(ext),
                body = (ctx, ext_2) -> truncate(tns, ctx, ext, ext_2)
            ),
            Phase(
                body = (ctx, ext_2) -> Run(Fill(literal(missing))),
            )
        ])
    )
end

struct Offset end

Base.show(io::IO, ex::Offset) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Offset)
	print(io, "Offset()")
end

const offset = Offset()

#Base.size(vec::Offset) = (nodim, nodim)

Base.getindex(arr::Offset, d, i) = d - i

struct VirtualOffset end

Base.show(io::IO, ex::VirtualOffset) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualOffset)
	print(io, "VirtualOffset()")
end

FinchNotation.finch_leaf(x::VirtualOffset) = virtual(x)

virtualize(ex, ::Type{Offset}, ctx) = VirtualOffset()

(ctx::LowerJulia)(tns::VirtualOffset) = :(Offset())

virtual_size(arr::VirtualOffset, ctx::LowerJulia, dim) = (nodim, nodim)
virtual_resize!(arr::VirtualOffset, ctx::LowerJulia, idx_dim, delta_dim) = (arr, virtual_eldim(arr, ctx, delta_dim, idx_dim))
virtual_eldim(arr::VirtualOffset, ctx::LowerJulia, idx_dim, delta_dim) = combinedim(ctx, widendim(shiftdim(idx_dim, call(-, getstop(delta_dim)))), widendim(idx_dim))

function get_reader(::VirtualOffset, ctx, proto_delta, proto_idx)
    tns = Furlable(
        size = (nodim, nodim),
        body = (ctx, ext) -> Lookup(
            body = (ctx, delta) -> Furlable(
                size = (nodim,),
                body = nothing,
                fuse = (tns, ctx, ext) -> Pipeline([
                    Phase(
                        stride = (ctx, ext_2) -> call(+, getstart(ext), delta, -1),
                        body = (ctx, ext_2) -> Run(Fill(literal(missing))),
                    ),
                    Phase(
                        stride = (ctx, ext_2) -> call(+, getstop(ext), delta),
                        body = (ctx, ext_2) -> truncate(Shift(tns, delta), ctx, ext, ext_2)
                    ),
                    Phase(
                        body = (ctx, ext_2) -> Run(Fill(literal(missing))),
                    )
                ])
            )
        )
    )
end

struct StaticOffset{Delta}
    delta::Delta
end

Base.show(io::IO, ex::StaticOffset) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::StaticOffset)
	print(io, "StaticOffset($(ex.delta))")
end

const staticoffset = StaticOffset

Base.getindex(arr::StaticOffset, i) = i - arr.delta

struct VirtualStaticOffset
    delta
end

Base.show(io::IO, ex::VirtualStaticOffset) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualStaticOffset)
	print(io, "VirtualStaticOffset($(ex.delta))")
end

FinchNotation.finch_leaf(x::VirtualStaticOffset) = virtual(x)

function virtualize(ex, ::Type{StaticOffset{Delta}}, ctx) where {Delta}
    VirtualStaticOffset(virtualize(:($ex.delta), Delta, ctx))
end

(ctx::LowerJulia)(tns::VirtualStaticOffset) = :(StaticOffset($tns.delta))

virtual_size(arr::VirtualStaticOffset, ctx::LowerJulia, dim = nodim) = (widendim(shiftdim(dim, getstop(arr.delta))),)
virtual_resize!(arr::VirtualStaticOffset, ctx::LowerJulia, idx_dim) = (arr, virtual_eldim(arr, ctx, idx_dim))
virtual_eldim(arr::VirtualStaticOffset, ctx::LowerJulia, idx_dim) = widendim(shiftdim(idx_dim, call(-, getstop(arr.delta))))

function get_reader(arr::VirtualStaticOffset, ctx, proto_idx)
    Furlable(
        size = (nodim,),
        body = nothing,
        fuse = (tns, ctx, ext) -> Pipeline([
            Phase(
                stride = (ctx, ext_2) -> call(+, getstart(ext), arr.delta, -1),
                body = (ctx, ext_2) -> Run(Fill(literal(missing))),
            ),
            Phase(
                stride = (ctx, ext_2) -> call(+, getstop(ext), arr.delta),
                body = (ctx, ext_2) -> truncate(Shift(tns, arr.delta), ctx, ext, ext_2)
            ),
            Phase(
                body = (ctx, ext_2) -> Run(Fill(literal(missing))),
            )
        ])
    )
end

struct Window{Start, Stop}
    start::Start
    stop::Stop
end

const window = Window

Base.show(io::IO, ex::Window) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Window)
	print(io, "Window(", ex.start, ", ", ex.stop, ")")
end

Base.size(vec::Window) = (vec.stop - vec.start + 1,)

function Base.getindex(arr::Window, i)
    vec.start + i - 1
end

struct VirtualWindow
    target
end

Base.show(io::IO, ex::VirtualWindow) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualWindow)
	print(io, "VirtualWindow(")
	print(io, ex.target)
	print(io, ")")
end

FinchNotation.finch_leaf(x::VirtualWindow) = virtual(x)

function virtualize(ex, ::Type{Window{Start, Stop}}, ctx) where {Start, Stop}
    start = virtualize(:($ex.start), Start, ctx)
    stop = virtualize(:($ex.stop), Stop, ctx)
    return VirtualWindow(Extent(start, stop))
end

(ctx::Finch.LowerJulia)(tns::VirtualWindow) = :(Window($(ctx(tns.target))))

virtual_size(arr::VirtualWindow, ctx::LowerJulia, dim) = (shiftdim(arr.target, call(-, getstart(dim), getstart(arr.target))),)
virtual_resize!(arr::VirtualWindow, ctx::LowerJulia, idx_dim) = (arr, arr.target)
virtual_eldim(arr::VirtualWindow, ctx::LowerJulia, idx_dim) = arr.target

function get_reader(arr::VirtualWindow, ctx, proto_idx)
    Furlable(
        size = (nodim,),
        body = nothing,
        fuse = (tns, ctx, ext) ->
            Shift(truncate(tns, ctx, ext, arr.target), call(-, getstart(ext), getstart(arr.target)))
    )
end