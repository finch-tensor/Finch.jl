using Finch
using Finch: Tensor, SubFiber, ElementLevel, DenseLevel, SparseListLevel, SparseCOOLevel

tikzshow(io, fbr::Tensor) = tikzshow(io, SubFiber(fbr.lvl, 1), "A", "A", 0, 0)
tikzshow(io, fbr) = tikzshow(io, fbr, 0)
tikzwidth(fbr::SubFiber{<:ElementLevel}) = 1
tikzwidth(fbr::SubFiber{<:DenseLevel}) =
    sum(p->max(tikzwidth(SubFiber(fbr.lvl.lvl, p)), 1), ((fbr.pos - 1) * fbr.lvl.shape) .+ (1:fbr.lvl.shape), init=1.5)
function tikzshow(io, fbr::SubFiber{<:DenseLevel}, tag, anchor, y0, x0)
    lvl = fbr.lvl
    mtx = "$(tag)d$(ndims(fbr))p$(fbr.pos)"
    lbl = "$(mtx)l"

    println(io, """
    \\node ($lbl) [anchor=north] at ($(x0 + tikzwidth(fbr)/2)*\\myunit, $y0*\\myunit) {pos=$(fbr.pos)};
    \\matrix ($mtx) [matrix of math nodes,
        nodes = {whclsty},
        left delimiter  = (,
        right delimiter = ),
        ampersand replacement=\\&,
        anchor=north] at ($(x0 + tikzwidth(fbr)/2)*\\myunit, $(y0-1)*\\myunit)
    {""")
    join(io, ["|[fullsty]|" for i=1:lvl.shape], "\\&"); println(io, "\\\\")
    println(io, "};")
    p = (fbr.pos - 1) * fbr.lvl.shape
    x = x0
    for i = 1:lvl.shape
        p += 1
        subfbr = SubFiber(lvl.lvl, p)
        subanchor = "$mtx-1-$i"
        subnode = tikzshow(io, subfbr, tag, subanchor, y0 - 4, x)
        println(io, "\\draw ($subanchor.center) -- ($subnode.north) node [midway, fill=white] {[$(":,"^(ndims(fbr)-1))$i]};")
        x += tikzwidth(subfbr)
    end
    lbl
end
function tikzwidth(fbr::SubFiber{<:SparseListLevel})
    lvl = fbr.lvl
    qoss = lvl.ptr[fbr.pos]:lvl.ptr[fbr.pos + 1]-1
    w = sum(p->max(tikzwidth(SubFiber(lvl.lvl, p)), 1), qoss, init=1.5)
    w += lvl.shape - length(qoss)
end

function tikzwidth(fbr::SubFiber{<:SparseCOOLevel})
    lvl = fbr.lvl
    qoss = lvl.ptr[fbr.pos]:lvl.ptr[fbr.pos + 1]-1
    w = sum(p->max(tikzwidth(SubFiber(lvl.lvl, p)), 1), qoss, init=1.5)
    w += *(lvl.shape...,) - length(qoss)
end

function tikzshow(io, fbr::SubFiber{<:SparseListLevel}, tag, anchor, y0, x0)
    lvl = fbr.lvl
    mtx = "$(tag)d$(ndims(fbr))p$(fbr.pos)"
    lbl = "$(mtx)l"
    qoss = lvl.ptr[fbr.pos]:lvl.ptr[fbr.pos + 1]-1

    println(io, """
    \\node ($lbl) [anchor=north] at ($(x0 + tikzwidth(fbr)/2)*\\myunit, $y0*\\myunit) {pos=$(fbr.pos)};
    \\matrix ($mtx) [matrix of math nodes,
        nodes = {whclsty},
        left delimiter  = (,
        right delimiter = ),
        ampersand replacement=\\&,
        anchor=north] at ($(x0 + tikzwidth(fbr)/2)*\\myunit, $(y0 - 1)*\\myunit)
    {""")
    join(io, [i in lvl.idx[qoss] ? "|[fullsty]|" : "|[zcsty]|" for i=1:lvl.shape], "\\&"); println(io, "\\\\")
    println(io, "};")
    #println(io, "\\node ($(node)l) [whclsty, anchor=south] at ($(x0 + tikzwidth(fbr)/2)*\\myunit, $y0*\\myunit) {};")
    #println(io, "\\node[whclsty, anchor=south] at ($(x0)*\\myunit, $y0*\\myunit) {$(lvl.ptr[fbr.pos])};")
    #println(io, "\\node[whclsty, anchor=south] at ($(x0 + tikzwidth(fbr))*\\myunit, $y0*\\myunit) {$(lvl.ptr[fbr.pos+1])};")
    x = x0
    for q in qoss
        i = lvl.idx[q]
        subfbr = SubFiber(lvl.lvl, q)
        subanchor = "$mtx-1-$i"
        subnode = tikzshow(io, subfbr, tag, subanchor, y0 - 4, x)
        println(io, "\\draw ($subanchor.center) -- ($subnode.north) node [midway, fill=white] {[$(":,"^(ndims(fbr)-1))$i]};")
        x += tikzwidth(subfbr)
    end
    #"$(node)l"
    lbl
end

function tikzshow(io, fbr::SubFiber{<:SparseCOOLevel{N}}, tag, anchor, y0, x0) where {N}
    lvl = fbr.lvl
    mtx = "$(tag)d$(ndims(fbr))p$(fbr.pos)"
    lbl = "$(mtx)l"
    qoss = lvl.ptr[fbr.pos]:lvl.ptr[fbr.pos + 1]-1

    println(io, """
    \\node ($lbl) [anchor=north] at ($(x0 + tikzwidth(fbr)/2)*\\myunit, $y0*\\myunit) {pos=$(fbr.pos)};
    \\matrix ($mtx) [matrix of math nodes,
        nodes = {whclsty},
        left delimiter  = (,
        right delimiter = ),
        ampersand replacement=\\&,
        anchor=north] at ($(x0 + tikzwidth(fbr)/2)*\\myunit, $(y0 - 1)*\\myunit)
    {""")
    idxs = collect(zip(lvl.tbl...))
    join(io, [Tuple(i) in idxs[qoss] ? "|[fullsty]|" : "|[zcsty]|" for i=CartesianIndices(lvl.shape)], "\\&"); println(io, "\\\\")
    println(io, "};")
    #println(io, "\\node ($(node)l) [whclsty, anchor=south] at ($(x0 + tikzwidth(fbr)/2)*\\myunit, $y0*\\myunit) {};")
    #println(io, "\\node[whclsty, anchor=south] at ($(x0)*\\myunit, $y0*\\myunit) {$(lvl.ptr[fbr.pos])};")
    #println(io, "\\node[whclsty, anchor=south] at ($(x0 + tikzwidth(fbr))*\\myunit, $y0*\\myunit) {$(lvl.ptr[fbr.pos+1])};")
    x = x0
    for q in qoss
        i = idxs[q]
        subfbr = SubFiber(lvl.lvl, q)
        subanchor = "$mtx-1-$(LinearIndices(fbr.lvl.shape)[i...])"
        subnode = tikzshow(io, subfbr, tag, subanchor, y0 - 4, x)
        println(io, "\\draw ($subanchor.center) -- ($subnode.north) node [midway, fill=white] {[$(":,"^(ndims(fbr)-N))$(join(i,","))]};")
        x += tikzwidth(subfbr)
    end
    #"$(node)l"
    lbl
end

function tikzshow(io, fbr::SubFiber{<:ElementLevel}, tag, anchor, y0, x0)
    lvl = fbr.lvl
    node = "$(tag)d$(ndims(fbr))p$(fbr.pos)"
    lbl = "$(node)l"
    println(io, "\\node ($lbl) [below=of $anchor] {\\\\$(fbr.pos)};")
    println(io, "\\node ($node) [below=0 of $(lbl).south, nzsty] {$(lvl.val[fbr.pos])};")
    return lbl
end

function tikzdisplay(f, name)
    open(name, "w") do io
        println(io, """
        \\documentclass{standalone}
        \\input{common.tex}
        \\begin{document}
        \\resizebox{\\linewidth}{!}{%
        \\begin{tikzpicture}[>=latex]
        """)
        f(io)
        println(io, """
        \\end{tikzpicture}%
        }
        \\end{document}
        """)
    end
end

highlight(io, fbr::Tensor) = highlight_level(io, fbr.lvl, "A", -1, tikzwidth(SubFiber(fbr.lvl, 1)), 0)

function highlight_level(io, lvl::DenseLevel, tag, x0, x1, y0)
    mtx = "$(tag)d$(Finch.level_ndims(typeof(lvl)))p1"
    lbl = "$(mtx)l"
    println(io, """
    \\draw [whclsty, anchor=north east] let \\p1 = ($(mtx)-1-1.north) in ($x0, \\y1) node {fibers:};
    \\draw [whclsty, anchor=north east] let \\p1 = ($(lbl).north) in ($x0, \\y1) node {DenseLevel positions:};
    """)
    highlight_level(io, lvl.lvl, tag, x0, x1, y0)
end

function highlight_level(io, lvl::SparseListLevel, tag, x0, x1, y0)
    mtx = "$(tag)d$(Finch.level_ndims(typeof(lvl)))p1"
    lbl = "$(mtx)l"
    #lbl_2 = "$(tag)d$(Finch.level_ndims(typeof(lvl)))p1l"
    println(io, """
    \\draw [whclsty, anchor=north east] let \\p1 = ($(mtx)-1-1.north) in ($x0, \\y1) node {fibers:};
    \\draw [whclsty, anchor=north east] let \\p1 = ($(lbl).north) in ($x0, \\y1) node {SparseLevel positions:};
    """)
    #\\draw [hlsty] let \\p1 = ($lbl.south), \\p2 = ($lbl.north) in ($x0*\\myunit, \\y1 - 1*\\myunit) rectangle ($x1*\\myunit, \\y2 - 1*\\myunit);
    #\\draw [hlsty] let \\p1 = ($lbl_2.south), \\p2 = ($lbl_2.north) in ($x0*\\myunit, \\y1) rectangle ($x1*\\myunit, \\y2);
    highlight_level(io, lvl.lvl, tag, x0, x1, y0)
end

function highlight_level(io, lvl::SparseCOOLevel{N}, tag, x0, x1, y0) where {N}
    mtx = "$(tag)d$(Finch.level_ndims(typeof(lvl)))p1"
    lbl = "$(mtx)l"
    #lbl_2 = "$(tag)d$(Finch.level_ndims(typeof(lvl)))p1l"
    println(io, """
    \\draw [whclsty, anchor=north east] let \\p1 = ($(mtx)-1-1.north) in ($x0, \\y1) node {fibers:};
    \\draw [whclsty, anchor=north east] let \\p1 = ($(lbl).north) in ($x0, \\y1) node {SparseCOO{$N} positions:};
    """)
    #\\draw [hlsty] let \\p1 = ($lbl.south), \\p2 = ($lbl.north) in ($x0*\\myunit, \\y1 - 1*\\myunit) rectangle ($x1*\\myunit, \\y2 - 1*\\myunit);
    #\\draw [hlsty] let \\p1 = ($lbl_2.south), \\p2 = ($lbl_2.north) in ($x0*\\myunit, \\y1) rectangle ($x1*\\myunit, \\y2);
    highlight_level(io, lvl.lvl, tag, x0, x1, y0)
end

function highlight_level(io, lvl::ElementLevel, tag, x0, x1, y0)
    lbl = "$(tag)d$(Finch.level_ndims(typeof(lvl)))p1"
    println(io, """
    \\draw [whclsty, anchor=north east] let \\p1 = ($(lbl).north) in ($x0, \\y1) node {values:};
    \\draw [whclsty, anchor=north east] let \\p1 = ($(lbl)l.north) in ($x0, \\y1) node {ElementLevel positions:};
    """)
    #\\draw [hlsty] let \\p1 = ($(lbl).south), \\p2 = ($lbl.north) in ($x0*\\myunit, \\y1) rectangle ($x1*\\myunit, \\y2);
end

A = [0 0 4.4;
    1.1 0 0;
    2.2 0 5.5;
    3.3 0 0]
(m, n) = size(A)

tikzdisplay("levels-A-d-d-e.tex") do io
    fbr = Tensor(Dense(Dense(Element(0.0))), A)
    tikzshow(io, fbr)
    highlight(io, fbr)
end

tikzdisplay("levels-A-d-sl-e.tex") do io
    fbr = Tensor(Dense(SparseList(Element(0.0))), A)
    tikzshow(io, fbr)
    highlight(io, fbr)
end

tikzdisplay("levels-A-sl-sl-e.tex") do io
    fbr = Tensor(SparseList(SparseList(Element(0.0))), A)
    tikzshow(io, fbr)
    highlight(io, fbr)
end

tikzdisplay("levels-A-sc2-e.tex") do io
    fbr = Tensor(SparseCOO{2}(Element(0.0)), A)
    tikzshow(io, fbr)
    highlight(io, fbr)
end

tikzdisplay("levels-A-matrix.tex") do io
    print(io, """
\\matrix (A) [matrix of math nodes,
  nodes = {whclsty},
  left delimiter  = (,
  right delimiter = ),
  ampersand replacement=\\&] at (0,0)
{
""")
    for i = 1:m
        print(io, "")
        join(io, (a == 0 ? "\\zecl" : "|[nzsty]|$a" for a in A[i, :]), " \\& ")
        println(io, "\\\\")
    end
    println(io, "};")
end