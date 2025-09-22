from sam.onyx.hw_nodes.hw_node import *
from lassen.utils import float2bfbin
import json


class ComputeNode(HWNode):
    def __init__(self, name=None, op=None, sam_graph_node_id=None,
                 mapped_coreir_dir=None, is_mapped_from_complex_op=False, original_complex_op_id=None) -> None:
        super().__init__(name=name)
        self.num_inputs = 2
        self.num_outputs = 1
        self.num_inputs_connected = 0
        self.num_outputs_connected = 0
        self.mapped_input_ports = []
        self.op = op
        self.opcode = None
        self.mapped_coreir_dir = mapped_coreir_dir
        # parse the mapped coreir file to get the input ports and opcode
        self.parse_mapped_json(self.mapped_coreir_dir + "/alu_coreir_spec_mapped.json",
                               sam_graph_node_id, is_mapped_from_complex_op, original_complex_op_id)
        assert self.opcode is not None

    def connect(self, other, edge, kwargs=None):

        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == ReadScannerNode:
            rd_scan = other.get_name()
            pe = self.get_name()
            new_conns = {
                'pe_to_rd_scan': [
                    ([(pe, "res"), (rd_scan, "us_pos_in")], 17),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            pe = self.get_name()
            new_conns = {
                'pe_to_wr_scan': [
                    ([(pe, "res"), (wr_scan, "data_in")], 17),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            in_str = "pos_in"
            offset = 1
            if 'crd' in edge.get_comment():
                in_str = "coord_in"
                offset = 0
            isect = other.get_name()
            pe = self.get_name()
            # isect_conn = other.get_num_inputs()

            if 'vector_reduce_mode' in edge.get_attributes():
                if edge.get_attributes()['vector_reduce_mode']:
                    isect_conn = 0
            else:
                if 'tensor' not in edge.get_attributes():
                    # Taking some liberties here - but technically this is the combo val
                    # isect_conn = other.get_connection_from_tensor('B')
                    isect_conn = other.get_connection_from_tensor('C')
                else:
                    isect_conn = other.get_connection_from_tensor(edge.get_tensor())

            new_conns = {
                f'pe_to_isect_{in_str}_{isect_conn}': [
                    ([(pe, "res"), (isect, f"{in_str}_{isect_conn}")], 17),
                ]
            }
            other.update_input_connections()
            return new_conns
        elif other_type == ReduceNode:
            other_red = other.get_name()
            pe = self.get_name()
            new_conns = {
                f'pe_to_reduce': [
                    ([(pe, "res"), (other_red, f"reduce_data_in")], 17),
                ]
            }
            return new_conns
        elif other_type == LookupNode:
            # TODO
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == MergeNode:
            # TODO
            # raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
            # Hack just use inner for now
            crddrop = other.get_name()
            pe = self.get_name()
            conn = 0
            if 'outer' in edge.get_comment():
                conn = 1

            new_conns = {
                f'pe_to_crddrop_res_to_{conn}': [
                    ([(pe, "res"), (crddrop, f"coord_in_{conn}")], 17),
                ]
            }
            return new_conns

        elif other_type == RepeatNode:
            # TODO
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == ComputeNode:
            other_pe = other.get_name()
            other_conn = other.get_num_inputs()
            pe = self.get_name()
            edge_attr = edge.get_attributes()
            # a destination port name has been specified by metamapper
            if "specified_port" in edge_attr and edge_attr["specified_port"] is not None:
                other_conn = edge_attr["specified_port"]
                other.mapped_input_ports.append(other_conn.strip("data"))
                new_conns = {
                    f'pe_to_pe_{other_conn}': [
                        ([(pe, "res"), (other_pe, f"{other_conn}")], 17),
                    ]
                }
            else:
                other_conn = other.mapped_input_ports[other_conn]
                new_conns = {
                    f'pe_to_pe_{other_conn}': [
                        ([(pe, "res"), (other_pe, f"data{other_conn}")], 17),
                    ]
                }
            other.update_input_connections()
            return new_conns
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == CrdHoldNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == FiberAccessNode:
            print("COMPUTE TO FIBER ACCESS")
            assert kwargs is not None
            assert 'flavor_that' in kwargs
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            print(kwargs)
            init_conns = self.connect(that_flavor, edge)
            print(init_conns)
            final_conns = other.remap_conns(init_conns, kwargs['flavor_that'])
            return final_conns
        else:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')

    def update_input_connections(self):
        self.num_inputs_connected += 1

    def get_num_inputs(self):
        return self.num_inputs_connected

    def parse_mapped_json(self, filename, node_id, is_mapped_from_complex_op, original_complex_op_id):
        with open(filename, 'r') as alu_mapped_file:
            alu_mapped = json.load(alu_mapped_file)
        # parse out the mapped opcode
        alu_instance_name = None
        module = None
        if not is_mapped_from_complex_op:
            module = alu_mapped["namespaces"]["global"]["modules"]["ALU_" + node_id + "_mapped"]
            for instance_name, instance in module["instances"].items():
                if "modref" in instance and instance["modref"] == "global.PE":
                    alu_instance_name = instance_name
                    break
            for connection in alu_mapped["namespaces"]["global"]["modules"]["ALU_" + node_id + "_mapped"]["connections"]:
                port0, port1 = connection
                if "self.in" in port0:
                    # get the port name of the alu
                    self.mapped_input_ports.append(port1.split(".")[1].strip("data"))
                elif "self.in" in port1:
                    self.mapped_input_ports.append(port0.split(".")[1].strip("data"))
            assert (len(self.mapped_input_ports) > 0)
        else:
            assert original_complex_op_id is not None
            module = alu_mapped["namespaces"]["global"]["modules"]["ALU_" + original_complex_op_id + "_mapped"]
            # node namae of a remapped alu node from a complex op is of the format <instance_name>_<id>
            alu_instance_name = '_'.join(node_id.split("_")[0:-1])
            # no need to find the input and output port for remapped op
            # as it is already assigned when we remap the complex op and stored in the edge object
        # look for the constant coreir object that supplies the opcode to the alu at question
        # insturction is supplied through the "inst" port of the alu
        for connection in module["connections"]:
            port0, port1 = connection
            if f"{alu_instance_name}.inst" in port0:
                constant_name = port1.split(".")[0]
            elif f"{alu_instance_name}.inst" in port1:
                constant_name = port0.split(".")[0]
        opcode = module["instances"][constant_name]["modargs"]["value"][1]
        opcode = "0x" + opcode.split('h')[1]

        self.opcode = int(opcode, 0)

    def configure(self, attributes):
        print("PE CONFIGURE")
        print(attributes)
        c_op = attributes['type'].strip('"')
        comment = attributes['comment'].strip('"')
        print(c_op)
        op_code = 0
        # configuring via sam, it is a sparse app
        use_dense = False
        # mapping to pe only, configuring only the pe, ignore the reduce
        pe_only = True
        # data I/O should interface with other primitive outside of the cluster
        pe_in_external = 1
        # according to the mapped input ports generate input port config
        num_sparse_inputs = list("000")
        for port in self.mapped_input_ports:
            num_sparse_inputs[2 - int(port)] = '1'
        print("".join(num_sparse_inputs))
        num_sparse_inputs = int("".join(num_sparse_inputs), 2)

        cfg_kwargs = {
            'op': self.opcode,
            'use_dense': use_dense,
            'pe_only': pe_only,
            'pe_in_external': pe_in_external,
            'num_sparse_inputs': num_sparse_inputs
        }
        return (op_code, use_dense, pe_only, pe_in_external, num_sparse_inputs), cfg_kwargs
