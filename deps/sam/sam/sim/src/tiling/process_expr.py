import argparse
import re
import ast
import copy

primitives = ["lvlscan", "union", "inter", "repeat", "lvlwr", "add", "mul", "crdhold", "reduce"]


def parse_file(args):
    filename = args.filename

    lines = None
    with open(filename, "r") as ff:
        lines = [line[:-1] for line in ff]
    assert lines is not None
    return lines


def permute(A, P, n):
    # For each element of P
    for i in range(n):
        next = i

        # Check if it is already
        # considered in cycle
        while (P[next] >= 0):
            # Swap the current element according
            # to the permutation in P
            t = A[i]
            A[i] = A[P[next]]
            A[P[next]] = t

            temp = P[next]

            # Subtract n from an entry in P
            # to make it negative which indicates
            # the corresponding move
            # has been performed
            P[next] -= n
            next = temp


def gen_crdhold_ast(expr, dictionary):
    [lhs, rhs] = expr.split("=")

    # Remove double quotes "..." from expr
    lhs = lhs[1:]
    rhs = rhs[:-1]

    tree = ast.parse(rhs)
    analyzer = CrdholdAnalyzer(dictionary)
    result = analyzer.visit(tree)
    return analyzer.needs_crdhold()


def update_dict(d1, d2):
    result = copy.deepcopy(d2)
    for k, v in d1.items():
        if k in d2:
            result[k] += v
        else:
            result[k] = v
    return result


def has_sparse_iteration(dictionary):
    result = []
    for k, v in dictionary.items():
        if len(v) > 1 and sum(fmt != 'd' for fmt in v) > 0:
            result.append(k)
    return result


def has_coiteration(dictionary):
    result = []
    for k, v in dictionary.items():
        if len(v) > 1 and sum(fmt != 'd' for fmt in v) > 1:
            result.append(k)
    return result


def has_outer(ind, coiter_ivars):
    sorted_ind = sorted(ind)
    assert coiter_ivars != []
    for ivar in coiter_ivars:
        assert ivar in sorted_ind, ivar + " not in " + str(sorted_ind)
        if sorted_ind.index(ivar) > 0:
            return True
    return False


class CrdholdAnalyzer(ast.NodeVisitor):
    def __init__(self, dictionary=None):
        self.tensor = None
        self.dictionary = dictionary
        self.tensors = []
        self.call = False
        self.outer_level_coiter = False

    def visit_Call(self, node):
        self.call = True
        self.tensor = node.func.id
        freevars = dict()
        for arg in node.args:
            ivar = self.visit(arg)
            freevars = update_dict(freevars, ivar)
        self.tensors.append(self.tensor)
        self.tensor = None
        self.call = False
        return freevars

    def visit_BinOp(self, node):

        freevars1 = self.visit(node.left)
        freevars2 = self.visit(node.right)

        freevars = update_dict(freevars1, freevars2)

        if isinstance(node.op, ast.Mult):
            iter_ivars = has_coiteration(freevars)
            ind = freevars.keys()
            if iter_ivars != [] and has_outer(ind, iter_ivars):
                self.outer_level_coiter = True

        return freevars

    def visit_UnaryOp(self, node):
        return self.visit(node.operand)

    def visit_Name(self, node):
        if self.call:
            assert self.tensor is not None
            ind = node.id
            if self.tensor not in self.tensors:
                d = self.dictionary[self.tensor]
            else:
                uid = get_num_repeats(self.tensors, self.dictionary, self.tensor)
                if uid > 0:
                    d = self.dictionary[self.tensor + str(uid)]
                else:
                    d = self.dictionary[self.tensor]
            if "lvl_format" in d:
                perm_formats = copy.deepcopy(d["lvl_format"])
                permute(perm_formats, [int(el) for el in d["perm"]], len(d["lvl_format"]))
                pos = d["ind"].index(ind)
                fmt = perm_formats[pos]
            else:
                fmt = 'd'
            out = {node.id: [fmt]}
        else:
            # Scalar
            out = {}
        return out

    def visit_Constant(self, node):
        return {}

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def needs_crdhold(self):
        return self.outer_level_coiter


def sort_num_tensors(lines):
    newlines = sorted(lines, key=lambda x: num_tensors(x[0]), reverse=True)
    for line in newlines:
        print(" ".join(line))
    return newlines


def num_tensors(expr):
    [lhs, rhs] = expr.split("=")

    # Remove double quotes "..." from expr
    lhs = lhs[1:]
    rhs = rhs[:-1]

    tree = ast.parse(rhs)
    analyzer = NumTensorAnalyzer()
    analyzer.visit(tree)
    return analyzer.get_num_tensors()


class NumTensorAnalyzer(ast.NodeVisitor):
    def __init__(self, dictionary=None):
        self.call = False
        self.num_tensors = 0

    def visit_Call(self, node):
        self.call = True
        self.num_tensors += 1
        for arg in node.args:
            ivar = self.visit(arg)
        self.call = False

    def visit_BinOp(self, node):

        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        self.visit(node.operand)

    def visit_Name(self, node):
        if not self.call:
            self.num_tensors += 1
            # Scalar

    def get_num_tensors(self):
        result = self.num_tensors
        self.num_tensors = 0
        return result


class FreeVarAnalyzer(ast.NodeVisitor):
    def __init__(self, dictionary=None):
        self.tensor = None
        self.dictionary = dictionary
        self.tensors = []
        self.call = False
        self.outer_level_coiter = False

    def visit_Call(self, node):
        self.call = True
        self.tensor = node.func.id
        freevars = []
        for arg in node.args:
            ivar = self.visit(arg)
            freevars += ivar
        self.tensors.append(self.tensor)
        self.tensor = None
        self.call = False
        return freevars

    def visit_BinOp(self, node):

        freevars1 = self.visit(node.left)
        freevars2 = self.visit(node.right)

        freevars = list(set(freevars1 + freevars2))
        return freevars

    def visit_UnaryOp(self, node):
        return self.visit(node.operand)

    def visit_Name(self, node):
        if self.call:
            assert self.tensor is not None
            ind = node.id
            if self.tensor not in self.tensors:
                d = self.dictionary[self.tensor]
            else:
                uid = get_num_repeats(self.tensors, self.dictionary, self.tensor)
                if uid > 0:
                    d = self.dictionary[self.tensor + str(uid)]
                else:
                    d = self.dictionary[self.tensor]
            if "lvl_format" in d:
                perm_formats = copy.deepcopy(d["lvl_format"])
                permute(perm_formats, [int(el) for el in d["perm"]], len(d["lvl_format"]))
                pos = d["ind"].index(ind)
                fmt = perm_formats[pos]
            else:
                fmt = 'd'
            out = [(node.id, fmt)]
        else:
            # Scalar
            out = []
        return out

    def visit_Constant(self, node):
        return []

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Module(self, node):
        return self.visit(node.body[0])


def gen_coiter_ast(expr, dictionary, is_dense=False):
    [lhs, rhs] = expr.split("=")

    # Remove double quotes "..." from expr
    lhs = lhs[1:]
    rhs = rhs[:-1]

    tree = ast.parse(rhs)
    analyzer = CoiterAnalyzer(dictionary, is_dense)
    result = analyzer.visit(tree)
    return analyzer.has_coiter()


class CoiterAnalyzer(ast.NodeVisitor):
    def __init__(self, dictionary=None, is_dense=False):
        self.tensor = None
        self.dictionary = dictionary
        self.tensors = []
        self.call = False
        self.coiter = False
        self.dense = is_dense

    def visit_Call(self, node):
        self.call = True
        self.tensor = node.func.id
        freevars = dict()
        for arg in node.args:
            ivar = self.visit(arg)
            freevars = update_dict(freevars, ivar)
        self.tensors.append(self.tensor)
        self.tensor = None
        self.call = False
        return freevars

    def visit_BinOp(self, node):

        freevars1 = self.visit(node.left)
        freevars2 = self.visit(node.right)

        freevars = update_dict(freevars1, freevars2)

        if isinstance(node.op, ast.Mult):
            if self.dense:
                iter_vars = has_sparse_iteration(freevars)
            else:
                iter_vars = has_coiteration(freevars)
            if iter_vars != []:
                self.coiter = True

        return freevars

    def visit_UnaryOp(self, node):
        return self.visit(node.operand)

    def visit_Name(self, node):
        if self.call:
            assert self.tensor is not None
            ind = node.id
            if self.tensor not in self.tensors:
                d = self.dictionary[self.tensor]
            else:
                uid = get_num_repeats(self.tensors, self.dictionary, self.tensor)
                if uid > 0:
                    d = self.dictionary[self.tensor + str(uid)]
                else:
                    d = self.dictionary[self.tensor]
            if "lvl_format" in d:
                perm_formats = copy.deepcopy(d["lvl_format"])
                permute(perm_formats, [int(el) for el in d["perm"]], len(d["lvl_format"]))
                pos = d["ind"].index(ind)
                fmt = perm_formats[pos]
            else:
                fmt = 'd'
            out = {node.id: [fmt]}
        else:
            # Scalar
            out = {}
        return out

    def visit_Constant(self, node):
        return {}

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def has_coiter(self):
        return self.coiter


def gen_union_ast(expr, dictionary, dense=False):
    [lhs, rhs] = expr.split("=")

    # Remove double quotes "..." from expr
    lhs = lhs[1:]
    rhs = rhs[:-1]

    tree = ast.parse(rhs)
    analyzer = UnionAnalyzer(dictionary, dense)
    result = analyzer.visit(tree)
    return analyzer.has_union()


class UnionAnalyzer(ast.NodeVisitor):
    def __init__(self, dictionary=None, dense=False):
        self.tensor = None
        self.dictionary = dictionary
        self.tensors = []
        self.dense = dense
        self.call = False
        self.union = False

    def visit_Call(self, node):
        self.call = True
        self.tensor = node.func.id
        freevars = dict()
        for arg in node.args:
            ivar = self.visit(arg)
            freevars = update_dict(freevars, ivar)
        self.tensors.append(self.tensor)
        self.tensor = None
        self.call = False
        return freevars

    def visit_BinOp(self, node):

        freevars1 = self.visit(node.left)
        freevars2 = self.visit(node.right)

        freevars = update_dict(freevars1, freevars2)

        if isinstance(node.op, ast.Add):
            iter_vars = has_sparse_iteration(freevars)
            if iter_vars != []:
                self.union = True

        return freevars

    def visit_UnaryOp(self, node):
        return self.visit(node.operand)

    def visit_Name(self, node):
        if self.call:
            assert self.tensor is not None
            ind = node.id
            if self.tensor not in self.tensors:
                d = self.dictionary[self.tensor]
            else:
                uid = get_num_repeats(self.tensors, self.dictionary, self.tensor)
                if uid > 0:
                    d = self.dictionary[self.tensor + str(uid)]
                else:
                    d = self.dictionary[self.tensor]
            if "lvl_format" in d:
                perm_formats = copy.deepcopy(d["lvl_format"])
                permute(perm_formats, [int(el) for el in d["perm"]], len(d["lvl_format"]))
                pos = d["ind"].index(ind)
                fmt = perm_formats[pos]
            else:
                fmt = 'd'
            out = {node.id: [fmt]}
        else:
            # Scalar
            out = {}
        return out

    def visit_Constant(self, node):
        return {}

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def has_union(self):
        return self.union


def remove_outer_parens(string):
    res = ""
    lpos = []
    rpos = []
    count = 0
    firstMatch = True

    for i, c in enumerate(string):
        if (c == '('):
            count += 1
            lpos.append(i)
            firstMatch = True
        elif (c == ')' and firstMatch):
            lpos.pop()
            firstMatch = False
        elif (c == ')'):
            firstMatch = False
            rpos.append(i)

    for i, c in enumerate(string):
        if i not in lpos and i not in rpos:
            res += c
    return res


def get_num_repeats(tensors, dictionary, tensorname):
    count = 0
    for tensor in tensors:
        if "orig_name" in dictionary[tensor]:
            name = dictionary[tensor]["orig_name"]
        else:
            name = tensor
        if tensorname == name:
            count += 1
    return count


def parse_tensors(expr, has_quotes=True):
    result = dict()

    [lhs, rhs] = expr.split("=")

    # Remove double quotes "..." from expr
    if has_quotes:
        lhs = lhs[1:]
        rhs = rhs[:-1]

    # Remove outer parenthesis from rhs
    rhs = remove_outer_parens(rhs)

    if "(" in lhs and ")" in lhs:
        # non-scalar
        lhs_tensor = lhs.split("(")[0]
        lhs_index_str = lhs.split("(")[1][:-1]
    else:
        # scalar
        lhs_tensor = lhs
        lhs_index_str = ''

    lhs_indices = lhs_index_str.split(",") if lhs_index_str != '' else []
    result["lhs_tensor"] = lhs_tensor
    result[lhs_tensor] = {"ind": lhs_indices}

    rhs_accesses = re.split(r'\*|\+|-|/', rhs)
    rhs_tensors = []
    for access in rhs_accesses:
        if "(" in access:
            # non-scalar
            tensor = access.split("(")[0]
            index_str = access.split("(")[1][:-1]
        else:
            tensor = access
            index_str = ''
        indices = index_str.split(",") if index_str != '' else []

        orig_tensor = tensor
        uid = get_num_repeats(rhs_tensors, result, tensor)
        if uid > 0:
            tensor = tensor + str(uid)

        rhs_tensors.append(tensor)
        result[tensor] = {"ind": indices, "orig_name": orig_tensor}

    result["rhs_tensors"] = rhs_tensors
    return result


def parse_all(line_args, has_quotes=True):
    expr = line_args[0]
    result = parse_tensors(expr, has_quotes)

    # Parse formats
    line_arg_idx = 1
    tensor = result["lhs_tensor"]
    if len(result[tensor]["ind"]) > 0 and len(line_args) > 1:
        # non-scalar
        assert tensor
        lhs_format = line_args[line_arg_idx]
        line_arg_idx += 1

        split = lhs_format.split(":")

        # Don't forget to remove the '-f='
        assert split[0][3:] == tensor, split[0][3:] + str(" != ") + tensor
        level_formats = split[1]
        level_formats = [*level_formats]

        # No permutations provided, assume default
        if len(split) > 2:
            permutation = split[2].split(",")
        else:
            permutation = list(range(len(level_formats)))
        result[tensor]["lvl_format"] = level_formats
        result[tensor]["perm"] = permutation

        for line in line_args[line_arg_idx:]:
            if "-f=" in line:
                rhs_format = line
                line_arg_idx += 1
                split = rhs_format.split(":")

                format_tensor = split[0][3:]

                level_formats = split[1]
                level_formats = [*level_formats]

                if len(split) > 2:
                    permutation = split[2].split(",")
                else:
                    permutation = list(range(len(level_formats)))

                for tensor in result["rhs_tensors"]:
                    orig_tensor = result[tensor]["orig_name"]
                    if orig_tensor == format_tensor:
                        result[tensor]["lvl_format"] = level_formats
                        result[tensor]["perm"] = permutation

    return result


def gen_unique_ast_fmt(line, dictionary):
    expr = line[0]

    [lhs, rhs] = expr.split("=")

    # Remove double quotes "..." from expr
    lhs = lhs[1:]
    rhs = rhs[:-1]

    rtree = ast.parse(rhs)
    ltree = ast.parse(lhs)
    rtransformer = TransformUniqueFmt(dictionary)
    rtransformer.reset()
    ast.fix_missing_locations(rtransformer.visit(rtree))

    ltransformer = TransformUniqueFmt(dictionary, varmap=rtransformer.varMap, t_uid=rtransformer.t_uid,
                                      iv_uid=rtransformer.iv_uid)
    ast.fix_missing_locations(ltransformer.visit(ltree))

    return (ltree, rtree)


#    varMap = ltransformer.varMap
#    new_dict = {}
#    lhs_tensor = dictionary["lhs_tensor"]
#    new_dict["lhs_tensor"] = varMap[lhs_tensor]
#    if lhs_tensor in dictionary:
#        new_dict[varMap[lhs_tensor]] = dictionary[lhs_tensor]
#
#    rhs_tensors = dictionary["rhs_tensors"]
#    new_rhs_tensors = [varMap[rhs_tensor] for rhs_tensor in rhs_tensors]
#    new_dict["rhs_tensors"] = new_rhs_tensors
#
#    for rhs_tensor in rhs_tensors:
#        new_dict[varMap[rhs_tensor]] = dictionary[rhs_tensor]
#
#
#    return (ltree, rtree), new_dict

class TransformUniqueFmt(ast.NodeTransformer):
    def __init__(self, dictionary, varmap={}, t_uid=0, iv_uid=0):
        self.dictionary = dictionary
        self.t_uid = t_uid
        self.iv_uid = iv_uid
        self.varMap = varmap
        self.call = False

    def reset(self):
        self.t_uid = 0
        self.iv_uid = 0
        self.varMap = {}
        self.call = False

    def visit_Call(self, node):
        self.call = True
        new_args = [self.visit(arg) for arg in node.args]
        self.call = False
        orig_tensor = node.func.id
        if orig_tensor in self.varMap:
            new_name = self.varMap[orig_tensor]
        else:
            new_name = 'a' + str(self.t_uid)
            if orig_tensor in self.dictionary:
                if 'lvl_format' in self.dictionary[orig_tensor]:
                    new_name += "_" + "".join(self.dictionary[orig_tensor]['lvl_format'])
                if 'perm' in self.dictionary[orig_tensor]:
                    new_name += "_" + "".join(self.dictionary[orig_tensor]['perm'])

            self.t_uid += 1
            self.varMap[orig_tensor] = new_name

        new_call = copy.deepcopy(node)
        new_call.func.id = new_name
        new_call.args = new_args
        return new_call

    def visit_Name(self, node):
        orig_name = node.id
        if orig_name in self.varMap:
            new_name = self.varMap[orig_name]
        else:
            if self.call:
                new_name = 'i' + str(self.iv_uid)
                self.iv_uid += 1
            else:
                new_name = 'a' + str(self.t_uid)
                self.t_uid += 1
            self.varMap[orig_name] = new_name
        new_node = copy.deepcopy(node)
        new_node.id = new_name
        return new_node


def gen_unique_ast(line):
    expr = line[0]

    [lhs, rhs] = expr.split("=")

    # Remove double quotes "..." from expr
    lhs = lhs[1:]
    rhs = rhs[:-1]

    rtree = ast.parse(rhs)
    ltree = ast.parse(lhs)

    rtransformer = TransformUnique()
    rtransformer.reset()
    ast.fix_missing_locations(rtransformer.visit(rtree))

    ltransformer = TransformUnique(varmap=rtransformer.varMap, t_uid=rtransformer.t_uid, iv_uid=rtransformer.iv_uid)
    ast.fix_missing_locations(ltransformer.visit(ltree))
    return (ltree, rtree)


class TransformUnique(ast.NodeTransformer):
    def __init__(self, varmap={}, t_uid=0, iv_uid=0):
        self.t_uid = t_uid
        self.iv_uid = iv_uid
        self.varMap = varmap
        self.call = False

    def reset(self):
        self.t_uid = 0
        self.iv_uid = 0
        self.varMap = {}
        self.call = False

    def visit_Call(self, node):
        self.call = True
        new_args = [self.visit(arg) for arg in node.args]
        self.call = False
        orig_tensor = node.func.id
        if orig_tensor in self.varMap:
            new_name = self.varMap[orig_tensor]
        else:
            new_name = 'a' + str(self.t_uid)
            self.t_uid += 1
            self.varMap[orig_tensor] = new_name

        new_call = copy.deepcopy(node)
        new_call.func.id = new_name
        new_call.args = new_args
        return new_call

    def visit_Name(self, node):
        orig_name = node.id
        if orig_name in self.varMap:
            new_name = self.varMap[orig_name]
        else:
            if self.call:
                new_name = 'i' + str(self.iv_uid)
                self.iv_uid += 1
            else:
                new_name = 'a' + str(self.t_uid)
                self.t_uid += 1
            self.varMap[orig_name] = new_name
        new_node = copy.deepcopy(node)
        new_node.id = new_name
        return new_node


def not_in1(tree, dictionary):
    for key in dictionary:
        #        if ast.unparse(key[0]) == ast.unparse(tree[0]) and ast.unparse(key[1]) == ast.unparse(tree[1]):
        if ast.dump(key[0]) == ast.dump(tree[0]) and ast.dump(key[1]) == ast.dump(tree[1]):
            return False
    return True


def not_in(tree, dictionary):
    for key in dictionary:
        if ast.unparse(key[0]) == ast.unparse(tree[0]) and ast.unparse(key[1]) == ast.unparse(tree[1]):
            #        if ast.dump(key[0]) == ast.dump(tree[0]) and ast.dump(key[1]) == ast.dump(tree[1]):
            return False
    return True


def uniqueify(lines, unique_fmt=True):
    trees = []
    dicts = []
    for line in lines:
        if unique_fmt:
            # Unique expressions
            dictionary = parse_all(line)
            tree = gen_unique_ast_fmt(line, dictionary)
        else:
            # Unique expressions only
            tree = gen_unique_ast(line)
        trees.append(tree)

    unique_dict = {}
    new_lines = []
    for i, tree in enumerate(trees):
        if not_in1(tree, unique_dict):
            print(" ".join(lines[i]))
            unique_dict[tree] = lines[i][0]

    print("ORIG LINES", len(lines))
    print("UNIQUE LINES", len(unique_dict.keys()))
    return lines


def clean_lines(lines):
    exprs = [line.split(" ")[3:] for line in lines]

    # Remove some cases that are ill-posed expressions in success.log
    exprs = [line for line in exprs if line[0] != '"' and '"struggle=ZGllKE' not in line[0]]

    return exprs


def expr_only_lines(lines):
    exprs = [line[0] for line in lines]
    return exprs


def find_add_mul(lines):
    result = []
    for i, line in enumerate(lines):
        if '*' in line and '+' in line:
            result.append(i)
    return result


def find_mul(lines):
    count = 0
    for line in lines:
        if '*' in line:
            count += 1
    return count


def find_add(lines):
    count = 0
    for line in lines:
        if '+' in line:
            count += 1
    return count


def only_dense(ll):
    dense = True
    for el in ll:
        if el != 'd':
            dense = False
    return dense


def find_output_dense(dictionary):
    count = 0
    for d in dictionary:
        tensor = d["lhs_tensor"]
        assert tensor in d, tensor + str(d)
        if "lvl_format" in d[tensor] and only_dense(d[tensor]["lvl_format"]):
            pass
            # print(d[tensor]["lvl_format"])
        if "lvl_format" not in d[tensor] or only_dense(d[tensor]["lvl_format"]):
            count += 1
    return count


def find_output_scalars(dictionary):
    count = 0
    for d in dictionary:
        tensor = d["lhs_tensor"]
        assert tensor in d, tensor + str(d)
        if len(d[tensor]["ind"]) == 0:
            count += 1
    return count


# The number of expressions that only have scalar inputs
def find_input_scalars(dictionary):
    count = 0
    for d in dictionary:
        all_scalars = True
        for tensor in d["rhs_tensors"]:
            if len(d[tensor]["ind"]) > 0:
                all_scalars = False

        if all_scalars:
            count += 1
    return count


# The number of expressions that only have dense inputs
def find_input_dense(dictionary):
    count = 0
    for d in dictionary:
        all_dense = True
        for tensor in d["rhs_tensors"]:
            if not ("lvl_format" in d[tensor] and only_dense(d[tensor]["lvl_format"])):
                all_dense = False

        if all_dense:
            count += 1
    return count


def find_broadcasts(dictionary, debug=False):
    count = 0
    for d in dictionary:
        lhs_tensor = d["lhs_tensor"]
        lhs_ind = d[lhs_tensor]["ind"]

        # idxVar on tensor order < max(rhs + lhs order) --> broadcast
        # idxVar on lhs not on rhs --> broadcast
        rhs_idx_set = set()
        rhs_idx_list = []
        for rhs_tensor in d["rhs_tensors"]:
            rhs_idx_set = rhs_idx_set.union(set(d[rhs_tensor]["ind"]))
            rhs_idx_list.append(d[rhs_tensor]["ind"])

        max_order = lhs_ind
        for rhs_tensor in d["rhs_tensors"]:
            if len(d[rhs_tensor]["ind"]) > len(max_order):
                max_order = d[rhs_tensor]["ind"]

        broadcast = False
        for rhs_tensor in d["rhs_tensors"]:
            for ind in max_order:
                if ind not in d[rhs_tensor]["ind"]:
                    broadcast = True

        for ind in lhs_ind:
            if ind not in rhs_idx_set:
                broadcast |= True

        if broadcast:
            count += 1
            if debug:
                print(lhs_ind, "?", rhs_idx_list)
        else:
            if debug:
                print("NO BROADCAST:", lhs_ind, "?", rhs_idx_list)
    return count


def find_reductions(dictionary, debug=False):
    count = 0
    for d in dictionary:
        lhs_tensor = d["lhs_tensor"]
        lhs_ind = d[lhs_tensor]["ind"]

        # idxVar on rhs != lhs --> reduction
        rhs_idx_set = set()
        rhs_idx_list = []
        for rhs_tensor in d["rhs_tensors"]:
            rhs_idx_set = rhs_idx_set.union(set(d[rhs_tensor]["ind"]))
            rhs_idx_list.append(d[rhs_tensor]["ind"])

        reduction = False
        for ind in rhs_idx_set:
            if ind not in lhs_ind:
                reduction = True
        reduction |= len(lhs_ind) < len(rhs_idx_set)

        if reduction:
            if debug:
                print(lhs_ind, "?", rhs_idx_set)
            count += 1
        else:
            if debug:
                print("NO REDUCTIONS:", lhs_ind, "?", rhs_idx_list)
    return count


def find_outer_level_mul():
    # assume topological ordering...
    pass


def find_expr(lines, primitive, is_dense=False, debug=False):
    assert primitive in primitives, primitive

    failed_lines = []
    dictionary = []
    count = 0
    if primitive == "lvlscan":
        for i, line in enumerate(lines):
            try:
                dictionary.append(parse_all(line))
            except KeyError:
                print(i, "FAILED")
                assert line[0] == '"' or '"struggle=ZGllKEBtZ' in line[0], line[0]
                failed_lines.append(i)

        if is_dense:
            # Uncompressed ONLY
            # separate them to uncompressed and compressed <-- process the format
            count = find_input_dense(dictionary)
        else:
            # scalars only
            count = find_input_scalars(dictionary)

    elif primitive == "union":
        # No elem-wise additions, reductions OK
        # Dense additions <-- process the format
        for i, line in enumerate(lines):
            try:
                dictionary = parse_all(line)
                has_sparse_mul = gen_union_ast(line[0], dictionary, is_dense)
                if debug:
                    if has_sparse_mul:
                        print("UNION:", i, line)
                    else:
                        print(i, line)
                count += int(has_sparse_mul)
            except KeyError:
                print(i, "FAILED")
                assert line[0] == '"' or '"struggle=ZGllKEBtZ' in line[0], line[0]
                failed_lines.append(i)

    elif primitive == "inter":
        # is_dense == True:
        # separate with and without locate
        # without locate, cannot do sparse iteration at all, fully dense only

        # is_dense == False:
        # No elem-wise multiplications of 2+ sparse levels, reductions OK
        # Dense multiplications OK
        # with locate
        for i, line in enumerate(lines):
            try:
                dictionary = parse_all(line)
                has_sparse_mul = gen_coiter_ast(line[0], dictionary, is_dense)
                if debug:
                    if has_sparse_mul:
                        if is_dense:
                            print("INT/LOC:", i, line)
                        else:
                            print("INTER :", i, line)
                    else:
                        print(i, line)
                count += int(has_sparse_mul)
            except KeyError:
                print(i, "FAILED")
                assert line[0] == '"' or '"struggle=ZGllKEBtZ' in line[0], line[0]
                failed_lines.append(i)

    elif primitive == "repeat":
        # No broadcasting, reductions OK
        # Only element-wise operations
        for i, line in enumerate(lines):
            try:
                dictionary.append(parse_all(line))
            except KeyError:
                print(i, "FAILED")
                assert line[0] == '"' or '"struggle=ZGllKEBtZ' in line[0], line[0]
                failed_lines.append(i)
        count = find_broadcasts(dictionary)

    elif primitive == "add":
        # assemble, multiplication, assignments

        # Addition count
        lines = expr_only_lines(lines)
        count = find_add(lines)
    elif primitive == "mul":
        # assemble, add, assignments

        # Multiplication count
        lines = expr_only_lines(lines)
        count = find_mul(lines)
    elif primitive == "reduce":
        # No reductions, broadcasts OK
        for i, line in enumerate(lines):
            try:
                dictionary.append(parse_all(line))
            except KeyError:
                print(i, "FAILED")
                assert line[0] == '"' or '"struggle=ZGllKEBtZ' in line[0], line[0]
                failed_lines.append(i)
        count = find_reductions(dictionary)

    elif primitive == "crdhold":
        # No multi-level multiplications (SpMSpV even not ok)
        # an intersection with any outer level above is not allowed.

        # FIXME: we need to consider output format here for recompression (?)
        for i, line in enumerate(lines):
            try:
                dictionary = parse_all(line)
                needs_crdhold = gen_crdhold_ast(line[0], dictionary)
                if debug:
                    if needs_crdhold:
                        print("CRDHOLD:", i, line)
                    else:
                        print(i, line)
                count += int(needs_crdhold)
            except KeyError:
                print(i, "FAILED")
                assert line[0] == '"' or '"struggle=ZGllKEBtZ' in line[0], line[0]
                failed_lines.append(i)

    elif primitive == "lvlwr":
        for i, line in enumerate(lines):
            try:
                dictionary.append(parse_all(line))
            except KeyError:
                print(i, "FAILED")
                assert line[0] == '"' or '"struggle=ZGllKEBtZ' in line[0], line[0]
                failed_lines.append(i)
        if is_dense:
            # If we keep the values level writer, we can do dense results.
            # Scalar and fully dense result count
            count = find_output_dense(dictionary)

        else:
            # No storage, scalar writeout only (?)
            # Scalar count
            count = find_output_scalars(dictionary)
    else:
        raise NotImplementedError

    if debug:
        print("failures", failed_lines)

    return count


def total_lines(lines):
    return len(lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process TACO expressions from website')
    parser.add_argument("--primitive", type=str, default="all")
    parser.add_argument("--filename", type=str, default="./success.log")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--unique", action="store_true")
    parser.add_argument("--uformat", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--sort", action="store_true")
    args = parser.parse_args()

    lines = parse_file(args)
    if args.clean:
        lines = clean_lines(lines)
    else:
        lines = [line.split(" ") for line in lines]

    if args.sort:
        sort_num_tensors(lines)
        exit()

    if args.unique:
        uniqueify(lines, args.uformat)
        exit()

    if args.count is not None:
        lines = lines[:args.count]

    total = len(lines)

    if args.primitive == "all":
        for primitive in primitives:
            count = find_expr(lines, primitive, args.dense, args.debug)
            print(primitive, "count", str(count), "percent", count / total * 100.0)
    else:
        primitive = args.primitive
        count = find_expr(lines, primitive, args.dense, args.debug)
        print(primitive, "count", str(count), "/", total, ",", "percent", count / total * 100.0)
