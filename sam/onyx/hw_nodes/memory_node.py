from sam.onyx.hw_nodes.hw_node import *


class MemoryNode(HWNode):
    def __init__(self, name=None) -> None:
        super().__init__(name=name)
        self._connected_to_buffet = False

    def connect(self, other, edge, kwargs=None):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode

        # Return false if already connected...
        if self._connected_to_buffet:
            return None
        new_conns = None
        other_type = type(other)

        this_name = self.get_name()
        other_name = other.get_name()

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == BuffetNode:
            buffet = other_name
            mem = this_name
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
            self._connected_to_buffet = True
            return new_conns
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == ReadScannerNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == WriteScannerNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == IntersectNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')
        elif other_type == CrdHoldNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        else:
            raise NotImplementedError(f'Cannot connect MemoryNode to {other_type}')

    def configure(self, attributes):
        return {"config": ["mek"], "mode": 'sram'}, {"config": ["mek"], "mode": 'ROM'}
