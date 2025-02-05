module FinchLogic
using RewriteTools, SyntaxInterface, AbstractTrees, Finch
using Finch: virtualize, freshen

export logic_leaf
export immediate
export deferred
export field
export alias
export table
export subquery
export query
export mapjoin
export aggregate
export reorder
export relabel
export reformat
export produces
export plan
export LogicNode

export isimmediate, isdeferred, isalias, isfield, isstateful
export getfields, getproductions, propagate_fields

export isprotocol

include("nodes.jl")
end
