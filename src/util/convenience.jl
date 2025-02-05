struct All{F}
    f::F
end

@inline (f::All{F})(args) where {F} = all(f.f, args)

struct Or{Fs}
    fs::Fs
end

Or(fs...) = Or{typeof(fs)}(fs)

@inline (f::Or{Fs})(arg) where {Fs} = any(g -> g(arg), f.fs)

struct And{Fs}
    fs::Fs
end

And(fs...) = And{typeof(fs)}(fs)

@inline (f::And{Fs})(arg) where {Fs} = all(g -> g(arg), f.fs)

kwfields(x::T) where {T} = Dict((k => getfield(x, k) for k in fieldnames(T))...)

(Base.:^)(T::Type, i::Int) = ∘(repeated(T, i)..., identity)
(Base.:^)(f::Function, i::Int) = ∘(repeated(f, i)..., identity)

pass_nothing(f, val) = val === nothing ? nothing : f(val)
