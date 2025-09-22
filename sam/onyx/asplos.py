import argparse
from sam.onyx.parse_dot import SAMDotGraph, parse_graph
import pydot

all_tests = [
    "mat_elemadd",
    'mat_elemadd3',
    'mat_elemmul',
    'mat_identity',
    'mat_mattransmul',
    'mat_residual',
    'mat_sddmm',
    'mat_vecmul_ij',
    'matmul_ijk',
    # 'matmul_jik',
    'tensor3_elemadd',
    'tensor3_elemmul',
    'tensor3_identity',
    'tensor3_innerprod',
    'tensor3_mttkrp',
    'tensor3_ttm',
    'tensor3_ttv',
    'vec_elemadd',
    'vec_elemmul',
    # 'vec_identity',
    # 'vec_scalar_mul',
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASPLOS argparser')
    parser.add_argument('--sam_graph',
                        type=str,
                        default="/home/max/Documents/SPARSE/sam/compiler/sam-outputs/dot/")
    parser.add_argument('--output_png',
                        type=str,
                        default="output.png")
    parser.add_argument('--output_graph',
                        type=str,
                        default="/home/max/Documents/SPARSE/sam/mek.gv")
    args = parser.parse_args()

    sam_graph = args.sam_graph
    output_png = args.output_png
    output_graph = args.output_graph

    fields = [
        'fiberlookup',
        'intersect',
        'repeat',
        'union',
        'add',
        'mul',
        'reduce',
        'crddrop',
        'fiberwrite',
        'arrayvals'
    ]

    fields_alt = [
        'app',
        'fiberlookup',
        'intersect',
        'repeat',
        'union',
        'alu',
        'reduce',
        'crddrop',
        'fiberwrite',
        'arrayvals'
    ]

    with open("results_block.csv", "w+") as output_csv:

        # output_csv.write("app,lvl_scan,repeat,intersect,union,alu,reduce,crd_drop,lvl_wr,arrays")
        output_csv.write(f"{','.join(fields_alt)}\n")

        for test_ in all_tests:
            test_path = f"{sam_graph}/{test_}.gv"
            # sdg = SAMDotGraph(filename=test_path, use_fork=True)
            graphs = pydot.graph_from_dot_file(test_path)
            graph = graphs[0]
            # graph = sdg.get_graph()
            # print(graph)
            type_cnt, hwnode_cnt = parse_graph(graph=graph)

            type_cnt_fnl = {}

            for type_, num in type_cnt.items():
                type_cnt_fnl[type_.strip('"')] = num

            # lvl_scan = type_cnt['fiberlookup'] if 'fiberlookup' in type_cnt else 0
            # repeat = type_cnt['repeat'] if 'repeat' in type_cnt else 0
            # intersect = type_cnt['intersect'] if 'intersect' in type_cnt else 0
            # union = type_cnt['union'] if 'union' in type_cnt else 0
            # add = type_cnt['add'] if 'add' in type_cnt else 0
            # mul = type_cnt['mul'] if 'mul' in type_cnt else 0
            # alu = add + mul
            # reduce = type_cnt['reduce'] if 'reduce' in type_cnt else 0
            # crddrop = type_cnt['crddrop'] if 'crddrop' in type_cnt else 0
            # lvl_wr = type_cnt['fiberwrite'] if 'fiberwrite' in type_cnt else 0
            # array = type_cnt['arrayvals'] if 'arrayvals' in type_cnt else 0
            fields_out = {}
            fields_out['app'] = test_

            for field_ in fields:
                # fields_out.append(str(type_cnt_fnl[field_] if field_ in type_cnt_fnl else 0))
                fields_out[field_] = str(type_cnt_fnl[field_] if field_ in type_cnt_fnl else 0)

            fields_out['alu'] = str(int(fields_out['add']) + int(fields_out['mul']))

            print(f"app: {test_}")
            print(type_cnt)
            print(type_cnt_fnl)
            print(fields_out)

            fields_out_find = []
            for field_ in fields_alt:
                fields_out_find.append(fields_out[field_])

            output_csv.write(f"{' & '.join(fields_out_find)}\n")
