abstract type FinchNodeInstance end

struct LiteralInstance{val} <: FinchNodeInstance end
struct IndexInstance{name} <: FinchNodeInstance end
struct VariableInstance{name} <: FinchNodeInstance end
struct DefineInstance{Lhs,Rhs,Body} <: FinchNodeInstance
    lhs::Lhs
    rhs::Rhs
    body::Body
end
struct DeclareInstance{Tns,Init,Op} <: FinchNodeInstance
    tns::Tns
    init::Init
    op::Op
end
struct FreezeInstance{Tns,Op} <: FinchNodeInstance
    tns::Tns
    op::Op
end
struct ThawInstance{Tns,Op} <: FinchNodeInstance
    tns::Tns
    op::Op
end
struct BlockInstance{Bodies} <: FinchNodeInstance
    bodies::Bodies
end
struct LoopInstance{Idx,Ext,Body} <: FinchNodeInstance
    idx::Idx
    ext::Ext
    body::Body
end
struct SieveInstance{Cond,Body} <: FinchNodeInstance
    cond::Cond
    body::Body
end
struct AssignInstance{Lhs,Op,Rhs} <: FinchNodeInstance
    lhs::Lhs
    op::Op
    rhs::Rhs
end
struct CallInstance{Op,Args<:Tuple} <: FinchNodeInstance
    op::Op
    args::Args
end
struct AccessInstance{Tns,Mode,Idxs} <: FinchNodeInstance
    tns::Tns
    mode::Mode
    idxs::Idxs
end
struct ReaderInstance{} <: FinchNodeInstance end
struct UpdaterInstance{Op} <: FinchNodeInstance
    op::Op
end
struct TagInstance{Var,Bind} <: FinchNodeInstance
    var::Var
    bind::Bind
end
struct YieldBindInstance{Args} <: FinchNodeInstance
    args::Args
end

function Base.getproperty(::LiteralInstance{val}, name::Symbol) where {val}
    name == :val ? val : error("type LiteralInstance has no field $name")
end
function Base.getproperty(::IndexInstance{val}, name::Symbol) where {val}
    name == :name ? val : error("type IndexInstance has no field $name")
end
function Base.getproperty(::VariableInstance{val}, name::Symbol) where {val}
    name == :name ? val : error("type VariableInstance has no field $name")
end

@inline literal_instance(val) = LiteralInstance{val}()
@inline index_instance(name) = IndexInstance{name}()
@inline variable_instance(name) = VariableInstance{name}()
@inline define_instance(lhs, rhs, body) = DefineInstance(lhs, rhs, body)
@inline declare_instance(tns, init, op) = DeclareInstance(tns, init, op)
@inline freeze_instance(tns, op) = FreezeInstance(tns, op)
@inline thaw_instance(tns, op) = ThawInstance(tns, op)
@inline block_instance(bodies...) = BlockInstance(bodies)
@inline loop_instance(idx, ext, body) = LoopInstance(idx, ext, body)
@inline loop_instance(body) = body
@inline sieve_instance(cond, body) = SieveInstance(cond, body)
@inline sieve_instance(body) = body
@inline sieve_instance(cond, args...) = SieveInstance(cond, sieve_instance(args...))
@inline assign_instance(lhs, op, rhs) = AssignInstance(lhs, op, rhs)
@inline call_instance(op, args...) = CallInstance(op, args)
@inline access_instance(tns, mode, idxs...) = AccessInstance(tns, mode, idxs)
@inline reader_instance() = ReaderInstance()
@inline updater_instance(op) = UpdaterInstance(op)
@inline tag_instance(var, bind) = TagInstance(var, bind)
@inline yieldbind_instance(args...) = YieldBindInstance(args)

@inline finch_leaf_instance(arg::Type) = literal_instance(arg)
@inline finch_leaf_instance(arg::Function) = literal_instance(arg)
@inline finch_leaf_instance(arg::FinchNodeInstance) = arg
@inline finch_leaf_instance(arg) = arg

SyntaxInterface.istree(node::FinchNodeInstance) = Int(operation(node)) & IS_TREE != 0
AbstractTrees.children(node::FinchNodeInstance) = istree(node) ? arguments(node) : []
isstateful(node::FinchNodeInstance) = Int(operation(node)) & IS_STATEFUL != 0

instance_ctrs = Dict(
    literal => literal_instance,
    index => index_instance,
    define => define_instance,
    declare => declare_instance,
    freeze => freeze_instance,
    thaw => thaw_instance,
    block => block_instance,
    loop => loop_instance,
    sieve => sieve_instance,
    assign => assign_instance,
    call => call_instance,
    access => access_instance,
    reader => reader_instance,
    updater => updater_instance,
    variable => variable_instance,
    tag => tag_instance,
    yieldbind => yieldbind_instance,
)

function SyntaxInterface.similarterm(::Type{FinchNodeInstance}, op::FinchNodeKind, args)
    instance_ctrs[op](args...)
end

SyntaxInterface.operation(::LiteralInstance) = literal
SyntaxInterface.operation(::IndexInstance) = index
SyntaxInterface.operation(::DefineInstance) = define
SyntaxInterface.operation(::DeclareInstance) = declare
SyntaxInterface.operation(::FreezeInstance) = freeze
SyntaxInterface.operation(::ThawInstance) = thaw
SyntaxInterface.operation(::BlockInstance) = block
SyntaxInterface.operation(::LoopInstance) = loop
SyntaxInterface.operation(::SieveInstance) = sieve
SyntaxInterface.operation(::AssignInstance) = assign
SyntaxInterface.operation(::CallInstance) = call
SyntaxInterface.operation(::AccessInstance) = access
SyntaxInterface.operation(::ReaderInstance) = reader
SyntaxInterface.operation(::UpdaterInstance) = updater
SyntaxInterface.operation(::VariableInstance) = variable
SyntaxInterface.operation(::TagInstance) = tag
SyntaxInterface.operation(::YieldBindInstance) = yieldbind

SyntaxInterface.arguments(node::DefineInstance) = [node.lhs, node.rhs, node.body]
SyntaxInterface.arguments(node::DeclareInstance) = [node.tns, node.init, node.op]
SyntaxInterface.arguments(node::FreezeInstance) = [node.tns]
SyntaxInterface.arguments(node::ThawInstance) = [node.tns]
SyntaxInterface.arguments(node::BlockInstance) = node.bodies
SyntaxInterface.arguments(node::LoopInstance) = [node.idx, node.ext, node.body]
SyntaxInterface.arguments(node::SieveInstance) = [node.cond, node.body]
SyntaxInterface.arguments(node::AssignInstance) = [node.lhs, node.op, node.rhs]
SyntaxInterface.arguments(node::CallInstance) = [node.op, node.args...]
SyntaxInterface.arguments(node::AccessInstance) = [node.tns, node.mode, node.idxs...]
SyntaxInterface.arguments(node::ReaderInstance) = []
SyntaxInterface.arguments(node::UpdaterInstance) = [node.op]
SyntaxInterface.arguments(node::TagInstance) = [node.var, node.bind]
SyntaxInterface.arguments(node::YieldBindInstance) = node.args

function Base.show(io::IO, node::LiteralInstance{val}) where {val}
    print(io, "literal_instance(", val, ")")
end
function Base.show(io::IO, node::IndexInstance{name}) where {name}
    print(io, "index_instance(", name, ")")
end
function Base.show(io::IO, node::VariableInstance{name}) where {name}
    print(io, "variable_instance(:", name, ")")
end
function Base.show(io::IO, node::FinchNodeInstance)
    print(io, instance_ctrs[operation(node)], "(")
    join(io, arguments(node), ", ")
    print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", node::FinchNodeInstance)
    print(io, "Finch program instance: ")
    if isstateful(node)
        display_statement(io, mime, node, 0)
    else
        display_expression(io, mime, node)
    end
end

Base.:(==)(a::VariableInstance, b::VariableInstance) = false
Base.:(==)(a::VariableInstance{tag}, b::VariableInstance{tag}) where {tag} = true
function Base.:(==)(a::FinchNodeInstance, b::FinchNodeInstance)
    return operation(a) == operation(b) && arguments(a) == arguments(b)
end
