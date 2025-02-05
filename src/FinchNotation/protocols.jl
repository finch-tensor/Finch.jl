isprotocol(f) = false

defaultread(i) = i
isprotocol(::typeof(defaultread)) = true

"""
    walk(i)
The walk protocol usually iterates over each pattern element of a tensor in
order. Note that the walk protocol "imposes" the structure of its argument on
the kernel, so that we specialize the kernel to the structure of the tensor.
"""
walk(i) = i
isprotocol(::typeof(walk)) = true

"""
    gallop(i)
The gallop protocol iterates over each pattern element of a tensor, leading the
iteration and superceding the priority of other tensors. Mutual leading is possible,
where we fast-forward to the largest step between either leader.
"""
gallop(i) = i
isprotocol(::typeof(gallop)) = true

"""
    follow(i)
The follow protocol ignores the structure of the tensor. By itself, the follow
protocol iterates over each value of the tensor in order, looking it up with
random access.  The follow protocol may specialize on e.g. the zero value of the
tensor, but does not specialize on the structure of the tensor. This enables efficient
random access and avoids large code sizes.
"""
follow(i) = i
isprotocol(::typeof(follow)) = true

defaultupdate(i) = i
isprotocol(::typeof(defaultupdate)) = true

"""
    laminate(i)
The laminate protocol declares that the tensor update may happen out of order and multiple times.
It is not usually necessary to declare a laminate protocol, but it is used internally to reason
about tensor format requirements.
"""
laminate(i) = i
isprotocol(::typeof(laminate)) = true

"""
    extrude(i)
The extrude protocol declares that the tensor update happens in order and only once, so that reduction
loops occur below the extrude loop. It is not usually necessary to declare an extrude protocol, but it
is used internally to reason about tensor format requirements.
"""
extrude(i) = i
isprotocol(::typeof(extrude)) = true
