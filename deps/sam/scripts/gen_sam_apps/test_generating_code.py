import pydot
import os
import argparse
import networkx as nx

from collections import defaultdict

frostt_list = ["tensor3_elemmul", "tensor3_identity", "tensor3_ttm", "tensor3_elemadd", "tensor3_innerprod",
               "tensor3_mttkrp", "tensor3_ttv", "tensor3_identity_dense"]
suitesparse_list = ["mat_elemmul", "mat_identity", "matmul_ijk", "matmul_ikj", "matmul_jki", "matmul_jik",
                    "matmul_kij", "matmul_jki", "mat_vecmul_ij", "mat_vecmul_ji", "matmul_kji",
                    "mat_elemadd3", "mat_sddmm.gv", "mat_elemadd", "mat_mattransmul",
                    "mat_residual", "mat_sddmm", "mat_identity_dense"]
vec_list = ["vec_elemadd", "vec_elemmul", "vec_scalar_mul", "vec_identity",
            "vec_scalar_mul", "vecmul", "vecmul_ij", "vecmul_ki"]
other_list = ["mat_mattransmul", "mat_residual", "tensor3_ttm", "tensor3_mttkrp", "tensor3_ttv", "mat_vecmul_ij",
              "mat_vecmul_ji"]


class TensorFormat:
    def __init__(self):
        self.tensors = {}

    def add_tensor_and_format(self, ten, form):
        self.tensors[ten] = form

    def set_format(self, ten, form):
        self.tensors[ten] = form

    def get_format(self, ten):
        return self.tensors[ten]

    def set_all_tensors(self, path):
        read_f = open(apath, "r")
        comment_line = read_f.readlines()
        comment = comment_line[1]
        comment = comment[comment.index("\"") + 1: -1]
        comment = comment[0: comment.index("\"")]
        comment = comment.split(",")
        for tensor_info in comment:
            node = tensor_info.split("=")
            self.add_tensor_and_format(node[0], node[1])
        return

    def return_size(self):
        return len(self.tensors)

    def return_all_tensors(self):
        return self.tensors.keys()

    def get_location(self, ten):
        if ten == "X" or ten == "x":
            return 0
        else:
            return 1


class CodeGenerationdatasets:
    def __init__(self):
        # Rememebers [parents of a node
        self.stream_join_elements = {}
        # ALl edges into a node
        self.edge_data = {}
        # Not used required since intersection has special structure need to know which reference is which coordinate
        self.intersect_dataset = defaultdict(dict)
        self.done_all = {}

    def build_datasets(self, networkx_graph):
        for u, v, a in networkx_graph.edges(data=True):
            if v not in self.done_all:
                self.done_all[v] = 0
            if v not in self.stream_join_elements:
                self.stream_join_elements[v] = [u]
                self.edge_data[v] = [str((a["label"]).strip('"'))]
            else:
                self.stream_join_elements[v].append(u)
                self.edge_data[v].append(str((a["label"]).strip('"')))

    def get_edge_data(self):
        return self.edge_data

    def get_edge_info(self, v, i):
        if self.edge_data[v][i] is None and len(self.edge_data[v][i]) == 0:
            return ""
        if "-i" in self.edge_data[v][i] or "-j" in self.edge_data[v][i] or "-k" in self.edge_data[v][i]:
            return str(self.edge_data[v][i])[:-2]
        return str(self.edge_data[v][i])

    def get_parents(self):
        return self.stream_join_elements

    def get_if_done(self):
        return self.done_all

    def add_done(self, a):
        self.done_all[a] = 1

    def set_edge_data(self, v, i, string):
        self.edge_data[v][i] = string

    def get_if_node_done(self, v):
        return self.done_all[v]


def tab(a):
    ans = ""
    for i in range(a):
        ans += "    "
    return ans


def sort_output_nodes(output_nodes, flag):
    node_with_vals = {}
    node_with_modes = {}
    for nodes in output_nodes.keys():
        if "vals" in output_nodes[nodes]:
            node_with_vals[nodes] = output_nodes[nodes]
        else:
            node_with_modes[nodes] = int(output_nodes[nodes])
    output = node_with_modes
    output = {}
    output = dict(sorted(node_with_modes.items(), key=lambda item: item[1]))
    flag = flag.replace("s", "")
    flag = flag.replace("d", "")
    flag = flag.replace("none", "")
    output2 = {}
    for i in flag:
        i = int(i)
        for key in output.keys():
            if output[key] == i:
                output2[key] = str(output[key])
                break

    for nodes in node_with_vals.keys():
        output2[nodes] = node_with_vals[nodes]
    return output2


def generate_header(f, out_name):
    f.write("import pytest\n")
    f.write("import time\n")
    f.write("import scipy.sparse\n")
    f.write("from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan\n")
    f.write("from sam.sim.src.wr_scanner import ValsWrScan\n")
    f.write("from sam.sim.src.joiner import Intersect2, Union2\n")
    f.write("from sam.sim.src.compute import Multiply2, Add2\n")
    f.write("from sam.sim.src.crd_manager import CrdDrop, CrdHold\n")
    f.write("from sam.sim.src.repeater import Repeat, RepeatSigGen\n")
    f.write("from sam.sim.src.accumulator import Reduce\n")
    f.write("from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2\n")
    f.write("from sam.sim.src.token import *\n")
    f.write("from sam.sim.test.test import *\n")
    f.write("from sam.sim.test.gold import *\n")
    f.write("from sam.sim.test.gen_gantt import gen_gantt\n")
    f.write("\n")
    f.write("import os\n")
    f.write("import csv\n")
    f.write("\n")
    f.write("cwd = os.getcwd()\n")
    if out_name in suitesparse_list:
        f.write("formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))\n")
    elif out_name in frostt_list:
        f.write("formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))\n")
    if out_name in other_list:
        f.write("other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))\n")

    f.write("\n\n# FIXME: Figureout formats\n")
    f.write("@pytest.mark.skipif(\n")
    f.write(tab(1) + "os.getenv('CI', 'false') == 'true',\n")
    f.write(tab(1) + "reason='CI lacks datasets',\n")
    f.write(")\n")
    if out_name in suitesparse_list:
        f.write("@pytest.mark.suitesparse\n")
    if out_name in frostt_list:
        f.write("@pytest.mark.frostt\n")
    if out_name in vec_list:
        f.write("@pytest.mark.vec\n")

    f.write("def test_" + out_name + "(samBench, " + get_dataset_name(out_name) + ", cast, check_gold, debug_sim, "
                                                                                  "report_stats, fill=0):\n")


def get_dataset_name(test_name):
    if test_name in frostt_list:
        return "frosttname"
    elif test_name in suitesparse_list:
        return "ssname"
    elif test_name in vec_list:
        return "vecname"
    else:
        return ""


def get_common_test_name(test_name):
    if "matmul" in test_name:
        return test_name[:-4]
    else:
        return test_name


def get_out_crd_str(d, u_, index_value):
    # By default, the input primitive connected to a crddrop will be a level scanner
    out_crd_str = "out_crd"
    # However, if the input primitive is another crddrop, we need to make sure it's reading from
    # the correct input crddrop output.
    if d[u_]["type"] == "crddrop":
        if index_value == d[u_]["inner"]:
            out_crd_str += "_inner"
        elif index_value == d[u_]["outer"]:
            out_crd_str += "_outer"
    return out_crd_str


def generate_datasets_code(f, tensor_formats, scope_lvl, tensor_info, tensor_format_parse, test_name):
    # Assuming the format is csr and csc:
    for ten in tensor_format_parse.return_all_tensors():
        if tensor_format_parse.get_location(ten) == 0:
            continue
        f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, " + get_dataset_name(test_name) +
                ", \"" + test_name + "\")\n")
        f.write(
            tab(scope_lvl) + ten + "_shape_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
            "_mode_shape\")\n")
        f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n")
        if tensor_format_parse.get_format(ten) == "ds01":
            f.write(
                tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname" + ",  \"tensor_" + ten +
                "_mode_1_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_dirname" + "tensor_" + ten + "_mode_1_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
        elif tensor_format_parse.get_format(ten) == "ds10":
            f.write(
                tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_0_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_0_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            # f.write(tab(scope_lvl) + ten + "_seg1" +  " = read_inputs(" + ten + "1_crd_filename)")
        elif tensor_format_parse.get_format(ten) == "ss01":
            f.write(
                tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname,  \"tensor_" + ten +
                "_mode_0_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_0_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            f.write(
                tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
        elif tensor_format_parse.get_format(ten) == "ss10":
            f.write(
                tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname,  \"tensor_" + ten +
                "_mode_0_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_0_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            f.write(
                tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")

        elif tensor_format_parse.get_format(ten) == "dss012":
            f.write(
                tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
            f.write(
                tab(scope_lvl) + ten + "2_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_2_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg2" + " = read_inputs(" + ten + "2_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "2_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_2_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd2" + " = read_inputs(" + ten + "2_crd_filename)\n\n")
        else:
            ct = 0
            for i in range(len(tensor_format_parse.get_format(ten))):
                if tensor_format_parse.get_format(ten)[i] == "s":
                    f.write(tab(scope_lvl) + ten + str(i) + "_seg_filename = os.path.join(" +
                            ten + "_dirname, \"tensor_" + ten +
                            "_mode_" + str(i) + "_seg\")\n")
                    f.write(
                        tab(scope_lvl) + ten + "_seg" + str(i) + " = read_inputs(" + ten + str(i) + "_seg_filename)\n")
                    f.write(tab(scope_lvl) + ten + str(i) + "_crd_filename = os.path.join(" +
                            ten + "_dirname, \"tensor_" + ten + "_mode_" + str(i) + "_crd\")\n")
                    f.write(tab(scope_lvl) + ten + "_crd" + str(i) + " = read_inputs(" + ten + str(
                        i) + "_crd_filename)\n\n")
        f.write(tab(scope_lvl) + ten + "_vals_filename = os.path.join(" + ten +
                "_dirname, \"tensor_" + ten + "_mode_vals\")\n")
        f.write(tab(scope_lvl) + ten + "_vals" + " = read_inputs(" + ten + "_vals_filename, float)\n\n")


def gen_data_formats(size, app_name, path):
    ans_list = []
    if size - 1 == 1:
        ans_list = ["orig"]
        return ans_list
    if size - 1 == 2:
        if "matmul" in app_name:
            ans_list = ["orig", "shift-trans"]
            return ans_list
        if "elemmul" in app_name:
            ans_list = ["orig", "shift"]
            return ans_list
        if "elemadd" in app_name:
            ans_list = ["orig", "shift"]
            return ans_list
        if "innerprod" in app_name:
            ans_list = ["orig", "shift"]
            return ans_list

        ans_list = ["orig", "other"]

        return ans_list
    else:
        ans_list = ["orig"]
        for i in range(size - 2):
            ans_list.append("other")
        return ans_list


def output_check_nodes(f, root_nodes):
    for r in root_nodes:
        f.write(tab(1) + "in_ref_" + str(r) + " = [0, 'D']\n")
    f.write(tab(1) + "done = False\n")
    f.write(tab(1) + "time_cnt = 0\n")


def finish_outputs(f, elements, nodes_completed):
    for i in nodes_completed:
        f.write(i)
    f.write("\n")
    output_list = ""
    # Write done statement
    f.write(tab(2) + "done = ")
    elements2 = []
    for elem in elements.keys():
        elements2.append(elem)
    for elem in elements.keys():
        f.write(elem + ".out_done()")
        if elem != elements2[-1]:
            f.write(" and ")
        else:
            f.write("\n")
    # TIme counter update
    f.write(tab(2) + "time_cnt += 1\n\n")
    # Autosize all blocks
    for elem in elements.keys():
        f.write(tab(1) + elem + ".autosize()\n")
    f.write("\n")

    if len(elements.keys()) > 1:
        f.write(tab(1) + "out_crds = [")
    else:
        f.write(tab(1) + "out_crds = []\n")

    output_list += " out_crds = ["
    count = 0
    for elem in elements.keys():
        if elements[elem] != "vals":
            f.write(elem + ".get_arr()")
            output_list += elem + ".get_arr()"
            count += 1
            if count < len(elements2) - 1:
                f.write(", ")
                output_list += ", "
            else:
                f.write("]\n")
                output_list += "]"
    count = 0
    if len(elements.keys()) > 1:
        f.write(tab(1) + "out_segs = [")
    else:
        f.write(tab(1) + "out_segs = []\n")

    output_list += ", out_segs = ["
    for elem in elements.keys():
        if elements[elem] != "vals":
            f.write(elem + ".get_seg_arr()")
            output_list += elem + ".get_seg_arr()"
            count += 1
            if count < len(elements2) - 1:
                f.write(", ")
                output_list += ", "
            else:
                f.write("]\n")
                output_list += "]"

    f.write(tab(1) + "out_vals = ")
    output_string = ", out_vals = "
    for elem in elements.keys():
        if elements[elem] == "vals":
            f.write(elem + ".get_arr()\n")
            output_string += elem + ".get_arr()"
    output_list += output_string
    return output_list


def generate_benchmarking_code(f, tensor_format_parse, test_name):
    f.write("\n")
    f.write(tab(1) + "# Print out cycle count for pytest output\n")
    f.write(tab(1) + "print(time_cnt)\n")
    f.write(tab(1) + "def bench():\n")
    f.write(tab(2) + "time.sleep(0.01)\n\n")
    f.write("\n")
    f.write(tab(1) + "extra_info = dict()\n")
    f.write(tab(1) + "extra_info[\"dataset\"] = " + get_dataset_name(test_name) + "\n")
    f.write(tab(1) + "extra_info[\"cycles\"] = time_cnt\n")
    ct = 0
    output_tensor = ""
    for k in tensor_format_parse.return_all_tensors():
        if ct == 0:
            output_tensor = k
        if ct != 0:
            f.write(tab(1) + "extra_info[\"tensor_" + k + "_shape\"] = " + k + "_shape\n")
        ct += 1
    statistic_available = ["fiberlookup", "reduce", "spaccumulator", "crddrop",
                           "repeat", "repeatsiggen", "intersect", "fiberwrite",
                           "arrayvals"]
    for u in networkx_graph.nodes():
        if d[u]["type"] in statistic_available:
            f.write(tab(1) + "sample_dict = " + d[u]["object"] + ".return_statistics()\n")
            f.write(tab(1) + "for k in sample_dict.keys():\n")
            f.write(tab(2) + "extra_info[\"" + d[u]["object"] + "\" + \"/\" + k] = sample_dict[k]\n\n")

    f.write(tab(1) + "gen_gantt(extra_info, \"" + test_name + "\")\n")
    f.write("\n")


def generate_check_against_gold_code(f, tensor_format_parse, test_name):
    f.write(tab(1) + "if check_gold:\n")
    f.write(tab(2) + "print(\"Checking gold...\")\n")
    f.write(tab(2) + "check_gold_")
    check = out_name[num]
    check = get_common_test_name(check)
    f.write(check + "(" + get_dataset_name(test_name) + ", debug_sim, cast, out_crds, out_segs, out_vals, \"" +
            tensor_format_parse.get_format(output_tensor) + "\")\n")
    f.write(tab(1) + "samBench(bench, extra_info)\n")


def size_computation_write(a):
    ans = " 1 "
    a = int(a)
    if a == 1:
        ans = " B_dim0 "
    return ans


def size_comp_write(a):
    ans = ""
    a = int(a)
    for i in range(a - 1):
        ans = " B_dim0 *"
    ans += "B_dim0"
    return ans


def breakup_node_info(node_name):
    d = dict(x.split("=") for x in node_name[1: -1].split(","))
    return d


def remove_broadcast_nodes(G):
    g = G.copy()
    for a in g:
        g0 = g.copy()
        node_i = breakup_node_info(g.nodes[a]["comment"])
        if node_i["type"] == "broadcast":
            for preds in g0.predecessors(a):
                for succs in g0.neighbors(a):
                    g0.add_edge(preds, succs, **(g0.get_edge_data(a, succs)[0]))
            g0.remove_node(a)
        g = g0
    return g


def parents_done(G, done, u):
    g = G.copy()
    ans = True
    for pred in G.predecessors(u):
        if pred not in done:
            ans = ans and True
        elif done[pred] == 1:
            ans = ans and True
        else:
            ans = False
    return ans


def array_size_computation(size_array):
    if "+" in size_array:
        size_arr = size_array.split("+")
    else:
        size_arr = [size_array]
    final_output_with_adds = ""
    output_array_with_adds = []
    for j in range(len(size_arr)):
        size_array = size_arr[j]
        if "*" in size_array:
            size_array = size_array.split("*")
        else:
            size_array = [size_array]
        output = []
        for strs in size_array:
            if len(strs) > 1:
                ans = strs[0]
                dim = strs[1]
                output.append(ans + "_shape[" + dim + "]")
            else:
                output.append(strs)
        output_str = ""
        for i in range(len(output) - 1):
            output_str += output[i]
            output_str += " * "
        output_str += output[-1]
        output_array_with_adds.append(output_str)
    for i in range(len(output_array_with_adds) - 1):
        final_output_with_adds += output_array_with_adds[i]
        final_output_with_adds += " + "
    final_output_with_adds += output_array_with_adds[-1]
    return final_output_with_adds


def get_all_files(directory_path):
    file_paths = []
    out_name = []

    for filename in os.listdir(directory_path):
        f = os.path.join(directory_path, filename)
        if filename[0] == ".":
            continue
        out_name.append(filename[0:-3])
        # checking if it is a file
        print("Test Name:", out_name[-1])
        if os.path.isfile(f):
            file_paths.append(f)
    return file_paths, out_name


parser = argparse.ArgumentParser("Generate sam apps/")
parser.add_argument("--input_dir", type=str, default="./compiler/sam-outputs/dot")
args = parser.parse_args()

file_paths, out_name = get_all_files(args.input_dir)
num = 0
for apath in file_paths:
    # apath = "../compiler/sam-outputs/dot/matmul_ijk.gv"
    # out_name[0] = "matmul_ijk"
    # apath = os.path.join(directory, ".gv")
    graphs = pydot.graph_from_dot_file(apath)
    graph = graphs[0]
    networkx_graph = nx.nx_pydot.from_pydot(graph)

    tensor_format_parse = TensorFormat()
    tensor_format_parse.set_all_tensors(apath)
    f = open(out_name[num] + ".py", "w")
    generate_header(f, out_name[num])
    # nx.draw(networkx_graph, with_labels=True)
    # plt.savefig("Graph.png", format="PNG")
    # plt.show()
    networkx_graph = remove_broadcast_nodes(networkx_graph)
    # Node Dataset present
    d = {}
    invalid_flag = 0
    root_nodes = []
    output_nodes = {}
    tensor_information = {}
    nodes_updating_list = []

    for u in list(nx.topological_sort(networkx_graph)):
        node_info = breakup_node_info(networkx_graph.nodes[u]["comment"])
        if node_info["type"] == "fiberlookup":
            if node_info["tensor"] not in tensor_information:
                tensor_information[node_info["tensor"]] = 0
            if node_info["format"] == "compressed":
                tensor_information[node_info["tensor"]] += 1 * (2 ** int(node_info["mode"]))
    tens_fmt = {}
    count = 0
    data_formats = gen_data_formats(len(tensor_format_parse.return_all_tensors()), out_name[num], apath)
    ct = 0
    for k in tensor_format_parse.return_all_tensors():
        if ct != 0:
            tens_fmt[k] = {}
            tens_fmt[k]["information"] = data_formats[ct - 1]
        ct += 1
    generate_datasets_code(f, tens_fmt, 1, tensor_information, tensor_format_parse, out_name[num])
    f.write("\n")

    for u in list(nx.topological_sort(networkx_graph)):
        node_info = breakup_node_info(networkx_graph.nodes[u]["comment"])
        d[u] = node_info
        if (node_info["type"] == "fiberlookup" or node_info["type"] == "repeat") and node_info["root"] == "true":
            root_nodes.append(node_info["tensor"])
        if node_info["type"] == "fiberlookup":
            if node_info["format"] == "dense":
                f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] +
                        node_info["index"] + "_" + str(u) + " = UncompressCrdRdScan(dim=" + node_info["tensor"] +
                        "_shape[" + node_info["mode"] + "]" + ", debug=debug_sim, statistics=report_stats)\n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u)

            if node_info["format"] == "compressed":
                f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + node_info["index"] +
                        "_" + str(u) + " = CompressedCrdRdScan(crd_arr=" + node_info["tensor"] +
                        "_crd" + node_info["mode"] + ", seg_arr=" + node_info["tensor"] +
                        "_seg" + node_info["mode"] + ", debug=debug_sim, statistics=report_stats)\n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u)

        elif node_info["type"] == "arrayvals":
            f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + "_" + str(u) + " = Array(init_arr=" +
                    node_info["tensor"] + "_vals, " + "debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + "_" + str(u)

        elif "broadcast" in networkx_graph.nodes[u]['comment']:
            continue

        elif node_info["type"] == "repsiggen":
            f.write(tab(1) + node_info["type"] + "_" + node_info["index"] + "_" + str(u) +
                    " = RepeatSigGen(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + "_" + node_info["index"] + "_" + str(u)

        elif node_info["type"] == "repeat":
            f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u) +
                    " = Repeat(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u)
        elif node_info["type"] == "intersect":
            f.write(tab(1) + node_info["type"] + node_info["index"] + "_" + str(u) + " = Intersect2(debug=debug_sim, " +
                    "statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + node_info["index"] + "_" + str(u)
        elif node_info["type"] == "union":
            f.write(tab(1) + node_info["type"] + node_info["index"] + "_" + str(u) + " = Union2(debug=debug_sim, " +
                    "statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + node_info["index"] + "_" + str(u)
            # invalid_flag = 1
        elif node_info["type"] == "spaccumulator" and node_info["order"] == "1":
            f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                u) + " = SparseAccumulator1(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + node_info["order"] + "_" + str(u)
            f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                u) + "_drop_crd_inner" + " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
            f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                u) + "_drop_crd_outer" + " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
            f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                u) + "_drop_val" + " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
        elif node_info["type"] == "spaccumulator" and node_info["order"] == "2":
            f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                u) + " = SparseAccumulator2(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + node_info["order"] + "_" + str(u)
            f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                u) + "_drop_crd_inner" + " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
            f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                u) + "_drop_crd_outer" + " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
            # f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
            #    u) + "_drop_crd_in_2" + " = StknDrop(debug=debug_sim)\n")
            f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                u) + "_drop_val" + " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
        elif node_info["type"] == "crddrop":
            f.write(
                tab(1) + node_info["type"] + "_" + str(u) + " = CrdDrop(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + "_" + str(u)
        elif node_info["type"] == "crdhold":
            f.write(
                tab(1) + node_info["type"] + "_" + str(u) + " = CrdHold(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + "_" + str(u)
        elif node_info["type"] == "mul":
            f.write(
                tab(1) + node_info["type"] + "_" + str(u) + " = Multiply2(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + "_" + str(u)
        elif node_info["type"] == "add":
            f.write(tab(1) + node_info["type"] + "_" + str(u) + " = Add2(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + "_" + str(u)
        elif node_info["type"] == "reduce":
            f.write(tab(1) + node_info["type"] + "_" + str(u) + " = Reduce(debug=debug_sim, statistics=report_stats)\n")
            d[u]["object"] = node_info["type"] + "_" + str(u)
        elif node_info["type"] == "fiberwrite":
            if node_info["mode"] == "vals":
                f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) +
                        " = ValsWrScan(size=" + array_size_computation(node_info["size"]) +
                        ", fill=fill, debug=debug_sim, statistics=report_stats)\n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u)
            elif node_info["format"] == "compressed":
                f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) +
                        " = CompressWrScan(seg_size=" + array_size_computation(node_info["segsize"]) + ", size=" +
                        array_size_computation(node_info["crdsize"]) + ", fill=fill," + " debug=debug_sim, " +
                        "statistics=report_stats)\n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u)
            else:
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u)
                continue
            if node_info["sink"] == "true":
                output_nodes[d[u]["object"]] = node_info["mode"]
        else:
            invalid_flag = 1
            print("Error invalid node detected", node_info["type"], "\n")
    if invalid_flag == 1:
        os.system("rm " + out_name[num] + ".py")
        print(out_name[num] + " failed\n")
        num += 1
        continue
    output_check_nodes(f, root_nodes)
    f.write("\n")
    f.write(tab(1) + "while not done and time_cnt < TIMEOUT:\n")

    intersect_dataset = defaultdict(dict)
    data = CodeGenerationdatasets()
    data.build_datasets(networkx_graph)
    for u in networkx_graph.nodes():
        if d[u]["type"] == "fiberlookup" and u not in data.get_if_done():
            if d[u]["root"] == "true":
                f.write(tab(2) + "if len(in_ref_" + d[u]["tensor"] + ") > 0:\n")
                f.write(tab(3) + d[u]["object"] + ".set_in_ref(in_ref_" + d[u]["tensor"] + ".pop(0))\n")
                nodes_updating_list.append(tab(2) + d[u]["object"] + ".update()\n\n")
                data.add_done(u)

    # FIXME: RENAME VARIABLE FROM i. Also figure out why this in range(10) is there...
    for i in range(10):
        for u, v, _ in list(nx.edge_bfs(networkx_graph)):  # .edges(data=True), networkx_graph.nodes())):
            a = networkx_graph.get_edge_data(u, v)[0]
            if d[v]["type"] == "fiberlookup" and data.get_if_node_done(v) == 0 and parents_done(networkx_graph,
                                                                                                data.get_if_done(), v):
                for i in range(len(data.get_parents()[v])):
                    u_ = data.get_parents()[v][i]
                    if "intersect" in d[u_]["object"] or "union" in d[u_]["object"]:
                        f.write(tab(2) + d[v]["object"] + ".set_in_ref(" + d[u_]["object"] + ".out_ref" +
                                str(intersect_dataset[d[u_]["object"]][d[v]["tensor"]]) + "())\n")
                    else:
                        f.write(tab(2) + d[v]["object"] + ".set_in_ref(" +
                                d[u_]["object"] + ".out_" +
                                str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "())\n")
                    nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

            if d[v]["type"] == "repsiggen" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for u_ in data.get_parents()[v]:
                    f.write(tab(2) + d[v]["object"] + ".set_istream(" + str(d[u_]["object"]).strip('"') +
                            ".out_" + str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "repeat" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                if d[v]["root"] == "true":
                    f.write(tab(2) + "if len(in_ref_" + d[v]["tensor"] + ") > 0:\n")
                    f.write(tab(3) + d[v]["object"] +
                            ".set_in_ref(in_ref_" + d[v]["tensor"] + ".pop(0))\n")
                for u_ in data.get_parents()[v]:
                    if "intersect" in d[u_]["object"] or "union" in d[u_]["object"]:
                        f.write(tab(2) + d[v]["object"] + ".set_in_ref(" +
                                d[u_]["object"] + ".out_ref" +
                                str(intersect_dataset[d[u_]["object"]][d[v]["tensor"]]) + "())\n")
                    else:
                        f.write(tab(2) + d[v]["object"] + ".set_in_" +
                                str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) +
                                "(" + d[u_]["object"] + ".out_" +
                                str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "arrayvals" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for u_ in data.get_parents()[v]:
                    if "intersect" in d[u_]["object"] or "union" in d[u_]["object"]:
                        f.write(tab(2) + d[v]["object"] + ".set_load(" + d[u_]["object"] + ".out_ref" +
                                str(intersect_dataset[d[u_]["object"]][d[v]["tensor"]]) + "())\n")
                    else:
                        f.write(tab(2) + d[v]["object"] + ".set_load(" + d[u_]["object"] + ".out_ref" + "())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "intersect" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for i in range(len(data.get_parents()[v])):
                    if i % 2 == 1:
                        continue
                    u_ = data.get_parents()[v][i]
                    f.write(tab(2) + d[v]["object"] + ".set_in" + str((i) // 2 + 1) + "(" +
                            d[u_]["object"] + ".out_ref(), " + d[u_]["object"] + ".out_crd())\n")
                    intersect_dataset[d[v]["object"]][d[u_]["tensor"]] = i // 2 + 1
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "union" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for i in range(len(data.get_parents()[v])):
                    if i % 2 == 1:
                        continue
                    u_ = data.get_parents()[v][i]
                    f.write(tab(2) + d[v]["object"] + ".set_in" + str((i) // 2 + 1) + "(" +
                            d[u_]["object"] + ".out_ref(), " + d[u_]["object"] + ".out_crd())\n")
                    intersect_dataset[d[v]["object"]][d[u_]["tensor"]] = i // 2 + 1
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "crddrop" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for u_ in data.get_parents()[v]:
                    index_value = data.get_edge_data()[v][data.get_parents()[v].index(u_)][-1]
                    if index_value == d[v]["inner"]:
                        out_crd_str = get_out_crd_str(d, u_, index_value)
                        f.write(tab(2) + d[v]["object"] + ".set_inner_crd" + "(" + d[u_]["object"] + "." +
                                out_crd_str + "())\n")
                    if index_value == d[v]["outer"]:
                        out_crd_str = get_out_crd_str(d, u_, index_value)
                        f.write(tab(2) + d[v]["object"] + ".set_outer_crd" + "(" + d[u_]["object"] + "." +
                                out_crd_str + "())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "crdhold" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for i in range(len(data.get_parents()[v])):
                    u_ = data.get_parents()[v][i]
                    index_value = data.get_edge_data()[v][i][-1]
                    local_edge = data.get_edge_data()[v][i][:-2]
                    if index_value == d[v]["inner"]:
                        f.write(tab(2) + d[v]["object"] + ".set_inner_crd" + "(" + d[u_]["object"] +
                                ".out_" + local_edge + "())\n")
                    if index_value == d[v]["outer"]:
                        f.write(tab(2) + d[v]["object"] + ".set_outer_crd" + "(" + d[u_]["object"] +
                                ".out_" + local_edge + "())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n")
                data.add_done(v)

            if d[v]["type"] == "spaccumulator" and d[v]["order"] == "1" and parents_done(networkx_graph,
                                                                                         data.get_if_done(), v) \
                    and data.get_if_node_done(v) == 0:
                for i in range(len(data.get_parents()[v])):
                    u_ = data.get_parents()[v][i]
                    local_edge = ""
                    if "crd" in data.get_edge_data()[v][i]:
                        local_edge = data.get_edge_data()[v][i][:-2]
                        f.write(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".set_in_stream(" +
                                d[u_]["object"] + ".out_" + local_edge + "())\n")
                    else:
                        local_edge = data.get_edge_data()[v][i]
                        f.write(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".set_in_stream(" +
                                d[u_]["object"] + ".out_val())\n")
                    nodes_updating_list.append(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".update()\n")

                for i in range(len(data.get_parents()[v])):
                    u_ = data.get_parents()[v][i]
                    local_edge = ""
                    if "crd" in data.get_edge_data()[v][i]:
                        local_edge = data.get_edge_data()[v][i][:-2]
                        f.write(tab(2) + d[v]["object"] + ".set_" + local_edge + "(" +
                                d[v]["object"] + "_drop_" + local_edge + ".out_val())\n")
                    else:
                        local_edge = data.get_edge_data()[v][i]
                        f.write(tab(2) + d[v]["object"] + ".set_" + local_edge + "(" +
                                d[v]["object"] + "_drop_" + local_edge + ".out_val())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "spaccumulator" and d[v]["order"] == "2" and parents_done(networkx_graph,
                                                                                         data.get_if_done(), v) \
                    and data.get_if_node_done(v) == 0:
                for i in range(len(data.get_parents()[v])):
                    u_ = data.get_parents()[v][i]
                    local_edge = ""
                    if "crd" in data.get_edge_data()[v][i]:
                        local_edge = data.get_edge_data()[v][i][:-2]
                        f.write(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".set_in_stream(" +
                                d[u_]["object"] + ".out_" + local_edge + "())\n")
                    else:
                        local_edge = data.get_edge_data()[v][i]
                        f.write(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".set_in_stream(" +
                                d[u_]["object"] + ".out_val())\n")
                    nodes_updating_list.append(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".update()\n")

                for i in range(len(data.get_parents()[v])):
                    u_ = data.get_parents()[v][i]
                    local_edge = ""
                    if "crd" in data.get_edge_data()[v][i]:
                        local_edge = data.get_edge_data()[v][i][:-2]
                        f.write(tab(2) + d[v]["object"] + ".set_" + local_edge + "(" +
                                d[v]["object"] + "_drop_" + local_edge + ".out_val())\n")
                    else:
                        local_edge = data.get_edge_data()[v][i]
                        f.write(tab(2) + d[v]["object"] + ".set_" + local_edge + "(" +
                                d[v]["object"] + "_drop_" + local_edge + ".out_val())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "mul" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for i in range(len(data.get_parents()[v])):
                    u_ = data.get_parents()[v][i]
                    f.write(tab(2) + d[v]["object"] + ".set_in" + str(data.get_parents()[v].index(u_) + 1) + "(" +
                            d[u_]["object"] + ".out_" + str(data.get_edge_info(v, i)) + "())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "add" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for u_ in data.get_parents()[v]:
                    f.write(tab(2) + d[v]["object"] + ".set_in" + str(data.get_parents()[v].index(u_) + 1) + "(" +
                            d[u_]["object"] + ".out_" + str(data.get_edge_info(v, i)) + "())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "reduce" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for u_ in data.get_parents()[v]:
                    f.write(tab(2) + d[v]["object"] + ".set_in_" +
                            str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "(" + d[u_]["object"] +
                            ".out_" + str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "())\n")
                nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                data.add_done(v)

            if d[v]["type"] == "fiberwrite" and parents_done(networkx_graph, data.get_if_done(), v) and \
                    data.get_if_node_done(v) == 0:
                for i in range(len(data.get_parents()[v])):
                    u_ = data.get_parents()[v][i]
                    if "val" not in data.get_edge_data()[v][i] and "spaccumulator" \
                            in d[u_]["object"]:
                        local_index = data.get_edge_data()[v][i][-1]
                        if d[u_]["in0"] == local_index:
                            local_cord = "_inner"
                        else:
                            local_cord = "_outer"
                        data.set_edge_data(v, i, "crd" + local_cord)
                    if d[v]["mode"] == "vals":
                        f.write(tab(2) + d[v]["object"] + ".set_input(" + d[u_]["object"] + ".out_" +
                                str(data.get_edge_info(v, i)) + "())\n")
                        nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                    else:
                        f.write(tab(2) + d[v]["object"] + ".set_input(" + d[u_]["object"] + ".out_" +
                                str(data.get_edge_info(v, i)) + "())\n")
                        nodes_updating_list.append(tab(2) + d[v]["object"] + ".update()\n")
                data.add_done(v)

    output_tensor = ""
    ct = 0
    for k in tensor_format_parse.return_all_tensors():
        if ct == 0:
            output_tensor = k
        ct += 1

    sorted_nodes = sort_output_nodes(output_nodes, tensor_format_parse.get_format(output_tensor))
    output_list = finish_outputs(f, sorted_nodes, nodes_updating_list)

    generate_benchmarking_code(f, tensor_format_parse, out_name[num])
    generate_check_against_gold_code(f, tensor_format_parse, out_name[num])

    f.close()
    os.system("cp " + out_name[num] + ".py ./sam/sim/test/apps/test_" + out_name[num] + ".py")
    os.system("rm " + out_name[num] + ".py")
    num += 1
