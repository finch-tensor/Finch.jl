from sam.onyx.parse_dot import *
import os


def combine_alu_ops(count_dict):

    add = 0
    mul = 0

    if 'add' in count_dict:
        add = count_dict['add']
        del count_dict['add']
    if 'mul' in count_dict:
        mul = count_dict['mul']
        del count_dict['mul']

    count_dict['ALU'] = add + mul


def clean_prim_count(count_dict):

    if 'broadcast' in count_dict:
        del count_dict['broadcast']

    if 'repsiggen' in count_dict:
        del count_dict['repsiggen']

    if 'fiberlookup' in count_dict:
        count_dict['Lvl Scan'] = count_dict['fiberlookup']
        del count_dict['fiberlookup']

    if 'fiberwrite' in count_dict:
        count_dict['Lvl Wr'] = count_dict['fiberwrite']
        del count_dict['fiberwrite']

    if 'arrayvals' in count_dict:
        count_dict['Array'] = count_dict['arrayvals']
        del count_dict['arrayvals']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SAM DOT Node Counting Script')
    parser.add_argument('--sam_graphs',
                        type=str,
                        default="/sam-artifact/sam/compiler/sam-outputs/dot/")
    parser.add_argument('--output_log',
                        type=str,
                        default="/sam-artifact/sam/tab1.log")

    args = parser.parse_args()

    graphs_to_count = {
        'mat_vecmul_ij': 'SpMV',
        'matmul_ijk': 'SpM*SpM (IJK Only)',
        'mat_sddmm': 'SDDMM',
        'tensor3_innerprod': 'InnerProd',
        'tensor3_ttv': 'TTV',
        'tensor3_ttm': 'TTM',
        'tensor3_mttkrp': 'MTTKRP',
        'mat_residual': 'Residual',
        'mat_mattransmul': 'MatTransMul',
        'mat_elemadd': 'MMAdd',
        'mat_elemadd3': 'Plus3',
        'tensor3_elemadd': 'Plus2',
    }

    figure_header = ['Lvl Scan', 'repeat', 'intersect', 'union', 'ALU', 'reduce', 'crddrop', 'Lvl Wr', 'Array']

    sam_graphs = args.sam_graphs
    nc_log = args.output_log

    with open(nc_log, 'w+') as nc_log_f_:

        for graph_, mapped_name in graphs_to_count.items():
            gv_path = os.path.join(sam_graphs, f"{graph_}.gv")
            graphs = pydot.graph_from_dot_file(gv_path)
            graph = graphs[0]

            prim_count = {}

            for node in graph.get_nodes():
                node_attr = node.get_attributes()
                type = node_attr['type'].strip('"')
                if type not in prim_count:
                    prim_count[type] = 1
                else:
                    prim_count[type] += 1

            clean_prim_count(prim_count)
            combine_alu_ops(prim_count)

            print(mapped_name)
            ps_ = ""
            for fh_ in figure_header:
                num = 0
                if fh_ in prim_count:
                    num = prim_count[fh_]
                ps_ += f"{fh_}: {num}\t"
            print(ps_)

            nc_log_f_.write(f"{mapped_name}\n")
            nc_log_f_.write(f"{ps_}\n")
