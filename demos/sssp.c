#include <julia.h>
#include "finch.h"
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
 

//  priorityQ[j][k] += (dist[k] > new_dist[k]) * (new_dist[k] == j) - (dist[k] > new_dist[k]) * (dist[k] == j)

 // new_priorityQ[j][k] = (dist[k] > new_dist[k]) * (new_dist[k] == j) + (dist[k] == new_dist[k] && j != priority) * priorityQ[j][k]
//  \forall k j priorityQ[j][k] += (dist[k] > new_dist[k]) * (new_dist[k] == j) - (dist[k] > new_dist[k]) * (dist[k] == j)
//  \forall k j priorityQ[j][k] += (mask[k] * (new_dist[k] == j) - mask[k]* (dist[k] == j))


JULIA_DEFINE_FAST_TLS // only define this once, in an executable

int P = 5;
int N = 5;
int source = 5;
jl_value_t* edges = 0;
jl_value_t* weights = 0;

struct sssp_data {
    jl_value_t* priorityQ;

    jl_value_t* dist;
};

//     priorityQ[p][j] = (p == 0 && j == source) + (p == (P - 1) && j != source)
jl_value_t* Init_priorityQ() {
    jl_function_t* pq_init = finch_eval("function pq_init(source, P, priorityQ)\n\
    @finch @loop p j priorityQ[p, j] = (p == 1 && j == $source) + (p == $P && j != $source)\n\
end");

    jl_value_t* priorityQ = finch_Fiber(
        finch_Dense(finch_Cint(P),
            finch_SparseList(finch_Cint(N),
                finch_ElementLevel(finch_Cint(0), finch_eval("Cint[]"))
            )
        )
    );

    finch_call(pq_init, finch_Cint(source), finch_Cint(P), priorityQ);
    printf("PQ init: ");
    finch_exec("println(%s.lvl.lvl.lvl.val)", priorityQ);
    return priorityQ;
}

//     dist[j] = (j != source) * P
jl_value_t* Init_dist() {
    jl_function_t* dist_init = finch_eval("function dist_init(source, P, dist)\n\
    @finch @loop j dist[j] = (j != $source) * $P\n\
end");

    jl_value_t* val = finch_eval("Cint[]");
    jl_value_t* dist = finch_Fiber(
        finch_Dense(finch_Cint(N),
        finch_ElementLevel(finch_Cint(0), val))
    );
    finch_call(dist_init, finch_Cint(source), finch_Cint(P), dist);
    printf("Dist init: ");
    finch_exec("println(%s.lvl.lvl.val)", dist);
    return dist;
}

// Let Init(source int) -> (dist float[N], priorityQ int[P][N])
//     dist[j] = (j != source) * P
//     priorityQ[p][j] = (p == 0 && j == source) + (p == (P - 1) && j != source)
// End
void Init(struct sssp_data* data) {
    jl_value_t* pq = Init_priorityQ();
    jl_value_t* d = Init_dist();

    data->priorityQ = pq;
    data->dist = d;
    
    return;
}

// Let UpdateEdges(dist float[N], priorityQ int[P][N], priority int) -> (new_dist float[N], new_priorityQ int[P][N], new_priority int)
//     new_dist[j] = edges[j][k] * priorityQ[priority][k] * (weights[j][k] + dist[k]) + (edges[j][k] * priorityQ[priority][k] == 0) * P | k:(MIN, dist[j])
//     new_priorityQ[j][k] = (dist[k] > new_dist[k]) * (j <= new_dist[k] &&  new_dist[k] < j + 1) + (dist[k] == new_dist[k] && j != priority) * priorityQ[j][k]
// 	new_priority = priority
// End
void UpdateEdges(struct sssp_data* old_data, struct sssp_data* new_data, int priority) {
    jl_function_t* new_dist_func = finch_eval("# new_dist[j] = edges[j][k] * priorityQ[priority][k] * (weights[j][k] + dist[k]) + (edges[j][k] * priorityQ[priority][k] == 0) * P | k:(MIN, dist[j])\n\
function new_dist_func(priority, N, P, new_dist, edges, priorityQ, weights, dist)\n\
    val = typemax(Cint)\n\
    B = Finch.Fiber(\n\
        Dense(N,\n\
            Element{val, Cint}([])\n\
        )\n\
    )\n\
    \n\
    @finch @loop p j k B[j] <<min>>= (p == $priority) * (edges[j, k] * priorityQ[p, k] * (weights[j, k] + dist[k]) + (edges[j, k] * priorityQ[p, k] == 0) * $P) + (p != $priority) * $val\n\
    @finch @loop j new_dist[j] = min(B[j], dist[j])\n\
end");
    jl_value_t* val = finch_eval("Cint[]");
    jl_value_t* new_dist = finch_Fiber(
        finch_Dense(finch_Cint(N),
        finch_ElementLevel(finch_Cint(0), val))
    );
    finch_call(new_dist_func, finch_Cint(priority), finch_Cint(N), finch_Cint(P), new_dist, edges, old_data->priorityQ, weights, old_data->dist);
    printf("New dist: \n");
    finch_exec("println(%s.lvl.lvl.val)", new_dist);

    jl_function_t* new_pq_func = finch_eval("function new_pq_func(new_priorityQ, old_priorityQ, dist, new_dist, priority)\n\
    @finch @loop j k new_priorityQ[j, k] = (dist[k] > new_dist[k]) * (new_dist[k] == j-1) + (dist[k] == new_dist[k] && j != $priority) * old_priorityQ[j, k]\n\
end");
    jl_value_t* val_pq = finch_eval("Cint[]");
    jl_value_t *new_priorityQ = finch_Fiber(
        finch_Dense(finch_Cint(P),
            finch_SparseList(finch_Cint(N),
                finch_ElementLevel(finch_Cint(0), finch_eval("Cint[]"))
            )
        )
    );
    finch_call(new_pq_func, new_priorityQ, old_data->priorityQ, old_data->dist, new_dist, finch_Cint(priority));
    printf("New pQ: \n");
    finch_exec("println(%s.lvl.lvl.lvl.val)", new_priorityQ);

    new_data->dist = new_dist;
    new_data->priorityQ = new_priorityQ;
}

// returns true if need to exit the loop
int inner_loop_condition(jl_value_t* priorityQ, int priority) {

    jl_value_t* access_func = finch_eval("function access_func(tensor1D, tensor2D, index)\n\
    @finch @loop j tensor1D[j] = tensor2D[$index, j]\n\
end");
    jl_value_t* val = finch_eval("Cint[]");
     jl_value_t* pq_slice = finch_Fiber(
        finch_SparseListLevel(finch_Cint(N), finch_eval("Cint[1, 1]"), finch_eval("Cint[]"),
        finch_ElementLevel(finch_Cint(0), finch_eval("Cint[]")))
    );
    finch_call(access_func, pq_slice, priorityQ, finch_Cint(priority));
    
    jl_value_t *pq_val = finch_exec("%s.lvl.lvl.val", pq_slice);
    double *pq_data = jl_array_data(pq_val);
    for(int i = 0; i < N; i++){
        if (pq_data[i] != 0) {
            return 0;
        }
    }

    return 1;   
}

// Let SSSP_one_priority_lvl(dist float[N], priorityQ int[P][N], priority int) -> (new_dist float[N], new_priorityQ int[P][N], new_priority int)
// 	new_dist, new_priorityQ, _ = UpdateEdges*(dist, priorityQ, priority) | (#2[#3] == 0)
//     new_priority = priority + 1
// End
int SSSP_one_priority_lvl(struct sssp_data* old_data, struct sssp_data* new_data, int priority) {

    new_data->dist = old_data->dist;
    new_data->priorityQ = old_data->priorityQ;

    while (inner_loop_condition(new_data->priorityQ, priority) == 0) {
        UpdateEdges(old_data, new_data, priority);
        old_data->dist = new_data->dist;
        old_data->priorityQ = new_data->priorityQ;
    }

    return priority + 1;
}

int outer_loop_condition(jl_value_t* priorityQ, int priority) {

    if (priority == P+1) {
        return 1;
    }

    jl_value_t *pq_val = finch_exec("%s.lvl.lvl.lvl.val", priorityQ);
    double *pq_data = jl_array_data(pq_val);
    for(int i = 0; i < N * P; i++){
        if (pq_data[i] != 0) {
            return 0;
        }
    }

    return 1;
}

// Let SSSP() -> (new_dist float[N], dist float[N], priorityQ int[P][N])
// 	dist, priorityQ = Init(source)
// 	new_dist, _, _ = SSSP_one_priority_lvl*(dist, priorityQ, 0) | (#2 == 0 || #3 == P)
// End
int SSSP(struct sssp_data* final_data) {
    struct sssp_data data_val = {};
    struct sssp_data *old_data = &data_val;
    Init(old_data);

    int priority = 1;

    while(outer_loop_condition(old_data->priorityQ, priority) == 0) {
        priority = SSSP_one_priority_lvl(old_data, final_data, priority);
        old_data->priorityQ = final_data->priorityQ;
        old_data->dist = final_data->dist;
    }

    return priority;
}


void setup1() {
    // 1 5, 4 5, 3 4, 2 3, 1 2
    // jl_value_t* edge_vector = finch_eval("Cint[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]");
    N = 5;
    source = 5;
    P = 5;
    edges = finch_eval("N = 5\n\
        edge_matrix = sparse([0 0 0 0 0; 1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 1 0 0 1 0])\n\
        Finch.Fiber(\n\
                 Dense(N,\n\
                 SparseList(N, edge_matrix.colptr, edge_matrix.rowval,\n\
                 Element{0.0}(edge_matrix.nzval))))");
    // 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
    weights = finch_eval("N = 4\n\
        matrix = sparse([0 0 0 0 0; 1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 3 0 0 1 0])\n\
        Finch.Fiber(\n\
                 Dense(N,\n\
                 SparseList(N, matrix.colptr, matrix.rowval,\n\
                 Element{0.0}(matrix.nzval))))");
    struct sssp_data d = {};
    struct sssp_data* data = &d;
    SSSP(data);

    printf("EXAMPLE1\nFinal: \n");
    finch_exec("println(%s.lvl.lvl.val)", data->dist);
}

void setup2() {
    // 2 1, 3 1, 3 2, 3 4
    N = 4;
    source = 1;
    P = 4;
    edges = finch_eval("N = 4\n\
        edge_matrix = sparse([0 1 1 0; 0 0 1 0; 0 0 0 0; 0 0 1 0])\n\
        Finch.Fiber(\n\
                 Dense(N,\n\
                 SparseList(N, edge_matrix.colptr, edge_matrix.rowval,\n\
                 Element{0.0}(edge_matrix.nzval))))");
    weights = finch_eval("N = 4\n\
        matrix = sparse([0 1 3 0; 0 0 1 0; 0 0 0 0; 0 0 5 0])\n\
        Finch.Fiber(\n\
                 Dense(N,\n\
                 SparseList(N, matrix.colptr, matrix.rowval,\n\
                 Element{0.0}(matrix.nzval))))");
    
    struct sssp_data d = {};
    struct sssp_data* data = &d;
    SSSP(data);

    printf("EXAMPLE2\nFinal: \n");
    finch_exec("println(%s.lvl.lvl.val)", data->dist);
}

void setup3() {
    // 2 1, 3 1, 1 2, 3 2, 1 3
    // jl_value_t* edge_vector = finch_eval("Cint[0, 1, 1, 1, 0, 0, 1, 1, 0]");
    N = 3;
    source = 1;
    P = 3;
    edges = finch_eval("N = 3\n\
        edge_matrix = sparse([0 1 1; 1 0 1; 1 0 0])\n\
        Finch.Fiber(\n\
                 Dense(N,\n\
                 SparseList(N, edge_matrix.colptr, edge_matrix.rowval,\n\
                 Element{0.0}(edge_matrix.nzval))))");
    weights = finch_eval("N = 4\n\
        matrix = sparse([0 1 5; 2 0 1; 1 0 0])\n\
        Finch.Fiber(\n\
                 Dense(N,\n\
                 SparseList(N, matrix.colptr, matrix.rowval,\n\
                 Element{0.0}(matrix.nzval))))");
    struct sssp_data d = {};
    struct sssp_data* data = &d;
    SSSP(data);

    printf("EXAMPLE3\nFinal: \n");
    finch_exec("println(%s.lvl.lvl.val)", data->dist);
}

void setup4() {
    // 2 3, 3 1, 4 3, 5 4, 6 5, 6 7, 7 5, 7 6
    // jl_value_t* edge_vector = finch_eval("Cint[0,0,0,0,0,0,0, 0,0,1,0,0,0,0, 1,0,0,0,0,0,0, 0,0,1,0,0,0,0, 0,0,0,1,0,0,0, 0,0,0,0,1,0,1, 0,0,0,0,1,1,0]");
    N = 7;
    P = 20;
    source = 1;
    edges = finch_eval("N = 7\n\
    edge_matrix = sparse([0 0 1 0 0 0 0; 0 0 0 0 0 0 0; 0 1 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 1 1; 0 0 0 0 0 0 1; 0 0 0 0 0 1 0])\n\
    Finch.Fiber(\n\
                Dense(N,\n\
                SparseList(N, edge_matrix.colptr, edge_matrix.rowval,\n\
                Element{0.0}(edge_matrix.nzval))))");
    weights = finch_eval("N = 4\n\
        matrix = sparse([0 0 1 0 0 0 0; 0 0 0 0 0 0 0; 0 9 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 3 1; 0 0 0 0 0 0 1; 0 0 0 0 0 1 0])\n\
        Finch.Fiber(\n\
                 Dense(N,\n\
                 SparseList(N, matrix.colptr, matrix.rowval,\n\
                 Element{0.0}(matrix.nzval))))");
    struct sssp_data d = {};
    struct sssp_data* data = &d;
    SSSP(data);

    printf("EXAMPLE4\n Final: \n");
    finch_exec("println(%s.lvl.lvl.val)", data->dist);
}

int main(int argc, char** argv) {
    finch_initialize();

    jl_value_t* res = finch_eval("using RewriteTools\n\
    using Finch.IndexNotation\n\
     using SparseArrays\n\
    ");

    res = finch_eval("or(x,y) = x == 1|| y == 1\n\
function choose(x, y)\n\
    if x != 0\n\
        return x\n\
    else\n\
        return y\n\
    end\n\
end");

    res = finch_eval("@slots a b c d e i j Finch.add_rules!([\n\
    (@rule @f(@chunk $i a (b[j...] <<min>>= $d)) => if Finch.isliteral(d) && i ∉ j\n\
        @f (b[j...] <<min>>= $d)\n\
    end),\n\
    (@rule @f(@chunk $i a @multi b... (c[j...] <<min>>= $d) e...) => begin\n\
        if Finch.isliteral(d) && i ∉ j\n\
            @f @multi (c[j...] <<min>>= $d) @chunk $i a @f(@multi b... e...)\n\
        end\n\
    end),\n\
    \n\
    (@rule @f($or(false, $a)) => a),\n\
    (@rule @f($or($a, false)) => a),\n\
    (@rule @f($or($a, true)) => true),\n\
    (@rule @f($or(true, $a)) => true),\n\
    \n\
    (@rule @f(@chunk $i a (b[j...] <<$choose>>= $d)) => if Finch.isliteral(d) && i ∉ j\n\
        @f (b[j...] <<$choose>>= $d)\n\
    end),\n\
    (@rule @f(@chunk $i a @multi b... (c[j...] <<choose>>= $d) e...) => begin\n\
        if Finch.isliteral(d) && i ∉ j\n\
            @f @multi (c[j...] <<choose>>= $d) @chunk $i a @f(@multi b... e...)\n\
        end\n\
    end),\n\
    (@rule @f($choose(0, $a)) => a),\n\
    (@rule @f($choose($a, 0)) => a),\n\
    (@rule @f(@chunk $i a (b[j...] <<$or>>= $d)) => if Finch.isliteral(d) && i ∉ j\n\
        @f (b[j...] <<$or>>= $d)\n\
    end),\n\
    (@rule @f(@chunk $i a @multi b... (c[j...] <<$or>>= $d) e...) => begin\n\
        if Finch.isliteral(d) && i ∉ j\n\
            @f @multi (c[j...] <<$or>>= $d) @chunk $i a @f(@multi b... e...)\n\
        end\n\
    end),\n\
])\n\
\n\
Finch.register()");
    setup1();

    setup2();

    setup3();

    setup4();

    finch_finalize();
}