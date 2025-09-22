from sam.onyx.hw_nodes.hw_node import *
from lake.modules.intersect import JoinerOp


class IntersectNode(HWNode):
    def __init__(self, name=None, conn_to_tensor=None) -> None:
        super().__init__(name=name)
        self.num_inputs = 2
        self.num_inputs_connected = 0
        self.num_outputs = 3
        self.num_outputs_connected = 0

        assert conn_to_tensor is not None
        self.conn_to_tensor = conn_to_tensor
        self.tensor_to_conn = {}
        for conn, tensor in self.conn_to_tensor.items():
            self.tensor_to_conn[tensor] = conn

    def connect(self, other, edge, kwargs=None):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

        new_conns = None
        isect = self.get_name()

        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == ReadScannerNode:
            rd_scan = other.get_name()
            comment = edge.get_attributes()['comment'].strip('"')
            try:
                tensor = comment.split("-")[1]
            except Exception:
                try:
                    tensor = comment.split("_")[1]
                except Exception:
                    tensor = comment
            out_conn = self.get_connection_from_tensor(tensor)

            new_conns = {
                f'isect_to_rd_scan': [
                    # send output to rd scanner
                    ([(isect, f"pos_out_{out_conn}"), (rd_scan, f"us_pos_in")], 17),
                    # ([(isect, f"eos_out_{1 + out_conn}"), (rd_scan, f"us_eos_in")], 1),
                    # ([(rd_scan, f"us_ready_out"), (isect, f"ready_in_{1 + out_conn}")], 1),
                    # ([(isect, f"valid_out_{1 + out_conn}"), (rd_scan, f"us_valid_in")], 1),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            comment = edge.get_attributes()['type'].strip('"')
            assert 'crd' in comment, f"isect to wrscan not crd type - is something up?"
            new_conns = {
                f'isect_to_wr_scan': [
                    # send output to rd scanner
                    ([(isect, f"coord_out"), (wr_scan, f"data_in")], 17),
                    # ([(isect, f"eos_out_0"), (wr_scan, f"eos_in_0")], 1),
                    # ([(wr_scan, f"ready_out_0"), (isect, f"ready_in_0")], 1),
                    # ([(isect, f"valid_out_0"), (wr_scan, f"valid_in_0")], 1),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            comment = edge.get_attributes()['comment'].strip('"')
            try:
                tensor = comment.split("-")[1]
            except Exception:
                try:
                    tensor = comment.split("_")[1]
                except Exception:
                    tensor = comment

            other_isect = other.get_name()
            isect_conn = self.get_connection_from_tensor(tensor)
            other_isect_conn = other.get_connection_from_tensor(tensor)

            edge_type = edge.get_attributes()['type'].strip('"')

            if 'crd' in edge_type:
                new_conns = {
                    f'isect_to_isect': [
                        # send output to rd scanner
                        ([(isect, f"coord_out"), (other_isect, f"coord_in_{other_isect_conn}")], 17),
                        # ([(isect, f"eos_out_0"), (wr_scan, f"eos_in_0")], 1),
                        # ([(wr_scan, f"ready_out_0"), (isect, f"ready_in_0")], 1),
                        # ([(isect, f"valid_out_0"), (wr_scan, f"valid_in_0")], 1),
                    ]
                }
            elif 'ref' in edge_type:
                new_conns = {
                    f'isect_to_isect': [
                        # send output to rd scanner
                        ([(isect, f"pos_out_{isect_conn}"), (other_isect, f"pos_in_{other_isect_conn}")], 17),
                        # ([(isect, f"eos_out_0"), (wr_scan, f"eos_in_0")], 1),
                        # ([(wr_scan, f"ready_out_0"), (isect, f"ready_in_0")], 1),
                        # ([(isect, f"valid_out_0"), (wr_scan, f"valid_in_0")], 1),
                    ]
                }
            return new_conns
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == MergeNode:
            print("INTERSECT TO MERGE")
            print(edge)
            merge = other.get_name()
            # Use inner to process outer
            merge_outer = other.get_outer()
            merge_inner = other.get_inner()
            conn = 0
            # print("INTERSECT TO MERGE")
            # print(edge)
            # print(edge.get_attributes())
            comment = edge.get_attributes()['comment'].strip('"')
            # print(comment)
            # print(merge_outer)
            # print(merge_inner)
            # okay this is dumb, stopgap until we can have super consistent output
            try:
                mapped_to_conn = comment.split("-")[1]
            except Exception:
                try:
                    mapped_to_conn = comment.split("_")[1]
                except Exception:
                    mapped_to_conn = comment
            if merge_outer in mapped_to_conn:
                conn = 1
            print(f"CONN: {conn}")
            new_conns = {
                f'isect_to_merger_{conn}': [
                    # Send isect row and isect col to merger inside isect_col
                    ([(isect, "coord_out"), (merge, f"coord_in_{conn}")], 17),
                ]
            }

            return new_conns
        elif other_type == RepeatNode:
            repeat = other.get_name()
            print("INTERSECT TO REPEAT EDGE!")
            out_conn = 0
            print(edge)
            comment = edge.get_attributes()['comment'].strip('"')
            cmt_tnsr = comment.split("-")[1]
            assert cmt_tnsr in self.tensor_to_conn
            out_conn = self.tensor_to_conn[cmt_tnsr]
            new_conns = {
                'intersect_to_repeat': [
                    # send output to rd scanner
                    ([(isect, f"pos_out_{out_conn}"), (repeat, "proc_data_in")], 17),
                ]
            }
            return new_conns
        elif other_type == ComputeNode:
            # Could be doing a sparse accum
            compute = other
            compute_name = other.get_name()
            edge_comment = edge.get_attributes()['comment'].strip('"')
            tensor = edge_comment.split('-')[1]
            out_conn = self.tensor_to_conn[tensor]
            compute_conn = compute.get_num_inputs()
            new_conns = {
                'intersect_to_repeat': [
                    # send output to rd scanner
                    ([(isect, f"pos_out_{out_conn}"), (compute_name, f"data{compute.mapped_input_ports[compute_conn]}")], 17),
                ]
            }
            compute.update_input_connections()
            return new_conns
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == RepSigGenNode:
            rsg = other.get_name()
            new_conns = {
                f'intersect_to_rsg': [
                    ([(isect, "coord_out"), (rsg, f"base_data_in")], 17),
                ]
            }
            return new_conns
        elif other_type == CrdHoldNode:
            print(edge)
            crdhold = other.get_name()
            edge_comment = edge.get_attributes()['comment'].strip('"')
            if 'outer' in edge_comment:
                conn = 1
            else:
                conn = 0
            new_conns = {
                f'intersect_to_crdhold': [
                    ([(isect, "coord_out"), (crdhold, f"cmrg_coord_in_{conn}")], 17),
                ]
            }
            return new_conns
        elif other_type == FiberAccessNode:
            print("INTERSECT TO FIBER ACCESS")
            assert kwargs is not None
            assert 'flavor_that' in kwargs
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            print(kwargs)
            init_conns = self.connect(that_flavor, edge)
            print(init_conns)
            final_conns = other.remap_conns(init_conns, kwargs['flavor_that'])
            return final_conns

        else:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')

        return new_conns

    def update_input_connections(self):
        self.num_inputs_connected += 1

    def get_num_inputs(self):
        return self.num_inputs_connected

    def get_connection_from_tensor(self, tensor):
        return self.tensor_to_conn[tensor]

    def get_tensor_from_connection(self, conn):
        return self.conn_to_tensor[conn]

    def configure(self, attributes):
        # print("INTERSECT CONFIGURE")
        # print(attributes)
        cmrg_enable = 0
        cmrg_stop_lvl = 0
        type_op = attributes['type'].strip('"')

        if 'vector_reduce_mode' in attributes:
            is_in_vr_mode = attributes['vector_reduce_mode'].strip('"')
            if is_in_vr_mode == "true":
                vr_mode = 1
        else:
            vr_mode = 0

        if type_op == "intersect":
            op = JoinerOp.INTERSECT.value
        elif type_op == "union":
            op = JoinerOp.UNION.value
        else:
            raise ValueError
        cfg_kwargs = {
            'cmrg_enable': cmrg_enable,
            'cmrg_stop_lvl': cmrg_stop_lvl,
            'op': op,
            'vr_mode': vr_mode
        }
        return (cmrg_enable, cmrg_stop_lvl, op, vr_mode), cfg_kwargs
