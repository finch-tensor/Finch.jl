from sam.onyx.hw_nodes.hw_node import *


class LookupNode(HWNode):
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
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode

        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == ReadScannerNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == WriteScannerNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == IntersectNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == CrdHoldNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        else:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')

    def configure(self, **kwargs):
        pass
