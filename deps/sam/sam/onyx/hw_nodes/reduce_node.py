from sam.onyx.hw_nodes.hw_node import *
from peak.assembler import Assembler
from hwtypes.modifiers import strip_modifiers
from lassen.sim import PE_fc as lassen_fc
import lassen.asm as asm


class ReduceNode(HWNode):
    def __init__(self, name=None) -> None:
        super().__init__(name=name)

    def connect(self, other, edge, kwargs=None):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

        red = self.get_name()

        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == ReadScannerNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            new_conns = {
                'reduce_to_wr_scan': [
                    # send output to rd scanner
                    ([(red, "reduce_data_out"), (wr_scan, "data_in")], 17),
                    # ([(red, "eos_out"), (wr_scan, "eos_in_0")], 1),
                    # ([(wr_scan, "ready_out_0"), (red, "ready_in")], 1),
                    # ([(red, "valid_out"), (wr_scan, "valid_in_0")], 1),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == ReduceNode:
            other_red = other.get_name()
            new_conns = {
                'reduce_to_reduce': [
                    # send output to rd scanner
                    ([(red, "reduce_data_out"), (other_red, "reduce_data_in")], 17),
                    # ([(red, "eos_out"), (wr_scan, "eos_in_0")], 1),
                    # ([(wr_scan, "ready_out_0"), (red, "ready_in")], 1),
                    # ([(red, "valid_out"), (wr_scan, "valid_in_0")], 1),
                ]
            }
            return new_conns
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == ComputeNode:
            pe = other.get_name()
            other_conn = other.mapped_input_ports[other.get_num_inputs()]
            new_conns = {
                f'reduce_to_pe_{other_conn}': [
                    ([(red, "reduce_data_out"), (pe, f"data{other_conn}")], 17),
                ]
            }
            other.update_input_connections()
            return new_conns
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')
        elif other_type == CrdHoldNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == FiberAccessNode:
            print("REDUCE TO FIBER ACCESS")
            assert kwargs is not None
            assert 'flavor_that' in kwargs
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            print(kwargs)
            init_conns = self.connect(that_flavor, edge)
            print(init_conns)
            final_conns = other.remap_conns(init_conns, kwargs['flavor_that'])
            return final_conns
        else:
            raise NotImplementedError(f'Cannot connect ReduceNode to {other_type}')

    def configure(self, attributes):
        # TODO
        stop_lvl = 2
        # bypassing the fifos in the pe, get result in a single cycle
        pe_only = False
        # configuring both the pe and the reduce
        pe_connected_to_reduce = True
        # data I/O to and from the PE should be internal with the reduce
        pe_in_external = 0
        # op is set to integer add for the PE TODO: make this configurable in the sam graph
        # TODO: make this use the metamapper
        instr_type = strip_modifiers(lassen_fc.Py.input_t.field_dict['inst'])
        asm_ = Assembler(instr_type)

        if 'fp' in attributes:
            is_fp = attributes['fp'].strip('"')
            if (is_fp == 'true'):
                op = int(asm_.assemble(asm.fp_add()))
            else:
                op = int(asm_.assemble(asm.add()))
        else:
            op = int(asm_.assemble(asm.add()))

        cfg_kwargs = {
            'stop_lvl': stop_lvl,
            'pe_connected_to_reduce': pe_connected_to_reduce,
            'pe_only': pe_only,
            'pe_in_external': pe_in_external,
            'op': op
        }
        return (stop_lvl, pe_connected_to_reduce, pe_only, pe_in_external, op), cfg_kwargs
