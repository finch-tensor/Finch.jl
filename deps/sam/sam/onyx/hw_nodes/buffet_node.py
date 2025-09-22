from sam.onyx.hw_nodes.hw_node import *
import kratos


class BuffetNode(HWNode):
    def __init__(self, name=None) -> None:
        super().__init__(name=name)

    def connect(self, other, edge, kwargs=None):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode

        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == MemoryNode:
            mem = other.get_name()
            buffet = self.get_name()
            new_conns = {
                'addr_to_mem': [
                    ([(buffet, "addr_to_mem"), (mem, "input_width_16_num_1"), (mem, "input_width_16_num_2")], 16),
                    ([(buffet, "data_to_mem"), (mem, "input_width_16_num_0")], 16),
                    ([(buffet, "wen_to_mem"), (mem, "input_width_1_num_1")], 1),
                    ([(buffet, "ren_to_mem"), (mem, "input_width_1_num_0")], 1),
                    ([(mem, "output_width_16_num_0"), (buffet, "data_from_mem")], 16),
                    ([(mem, "output_width_1_num_1"), (buffet, "valid_from_mem")], 1),
                    ([(mem, "output_width_1_num_0"), (buffet, "ready_from_mem")], 1),
                ]
            }
            return new_conns
        elif other_type == ReadScannerNode:
            rd_scan = other.get_name()
            buffet = self.get_name()
            new_conns = {
                'buffet_to_rd_scan': [
                    # rd rsp
                    ([(buffet, "rd_rsp_data"), (rd_scan, "rd_rsp_data_in")], 17),
                    # ([(rd_scan, "rd_rsp_ready_out"), (buffet, "rd_rsp_ready")], 1),
                    # ([(buffet, "rd_rsp_valid"), (rd_scan, "rd_rsp_valid_in")], 1),
                    # addr
                    ([(rd_scan, "addr_out"), (buffet, "rd_addr")], 17),
                    # ([(buffet, "rd_addr_ready"), (rd_scan, "addr_out_ready_in")], 1),
                    # ([(rd_scan, "addr_out_valid_out"), (buffet, "rd_addr_valid")], 1),

                    # op
                    ([(rd_scan, "op_out"), (buffet, "rd_op")], 17),
                    # ([(buffet, "rd_op_ready"), (rd_scan, "op_out_ready_in")], 1),
                    # ([(rd_scan, "op_out_valid_out"), (buffet, "rd_op_valid")], 1),

                    # id
                    ([(rd_scan, "ID_out"), (buffet, "rd_ID")], 17),
                    # ([(buffet, "rd_ID_ready"), (rd_scan, "ID_out_ready_in")], 1),
                    # ([(rd_scan, "ID_out_valid_out"), (buffet, "rd_ID_valid")], 1),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            buffet = self.get_name()
            new_conns = {
                'buffet_to_wr_scan': [
                    # wr op/data
                    ([(wr_scan, "data_out"), (buffet, "wr_data")], 17),
                    # ([(wr_scan, "op_out"), (buffet, "wr_op")], 1),
                    # ([(buffet, "wr_data_ready"), (wr_scan, "data_out_ready_in")], 1),
                    # ([(wr_scan, "data_out_valid_out"), (buffet, "wr_data_valid")], 1),
                    # addr
                    ([(wr_scan, "addr_out"), (buffet, "wr_addr")], 17),
                    # ([(buffet, "wr_addr_ready"), (wr_scan, "addr_out_ready_in")], 1),
                    # ([(wr_scan, "addr_out_valid_out"), (buffet, "wr_addr_valid")], 1),

                    # id
                    ([(wr_scan, "ID_out"), (buffet, "wr_ID")], 17),
                    # ([(buffet, "wr_ID_ready"), (wr_scan, "ID_out_ready_in")], 1),
                    # ([(wr_scan, "ID_out_valid_out"), (buffet, "wr_ID_valid")], 1),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')
        elif other_type == CrdHoldNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        else:
            raise NotImplementedError(f'Cannot connect BuffetNode to {other_type}')

    def configure(self, attributes):

        total_cap = 2048

        capacity_0 = 1024
        capacity_1 = total_cap - capacity_0

        fetch_width_log = 2

        cap0 = kratos.clog2(capacity_0) - fetch_width_log
        cap1 = kratos.clog2(capacity_1) - fetch_width_log

        if 'vector_reduce_mode' in attributes:
            is_in_vr_mode = attributes['vector_reduce_mode'].strip('"')
            if is_in_vr_mode == "true":
                vr_mode = 1
        else:
            vr_mode = 0

        cfg_kwargs = {
            'capacity_0': cap0,
            'capacity_1': cap1,
            'vr_mode': vr_mode
        }
        return (capacity_0, capacity_1, vr_mode), cfg_kwargs
