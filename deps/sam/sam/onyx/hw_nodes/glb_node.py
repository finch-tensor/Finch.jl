from sam.onyx.hw_nodes.hw_node import *


class GLBNode(HWNode):
    def __init__(self, name=None, data=None, valid=None, ready=None,
                 direction=None, num_blocks=None, seg_mode=None, file_number=None, tx_size=None, IO_id=0,
                 bespoke=False, tensor=None, mode=None, format=None) -> None:
        super().__init__(name=name)

        self.data = data
        self.valid = valid
        self.ready = ready
        self.direction = direction
        self.num_blocks = num_blocks
        self.seg_mode = seg_mode
        self.file_number = file_number
        self.tx_size = tx_size
        self.IO_id = IO_id
        self.tensor = tensor
        self.mode = mode
        self.format = format
        # If bespoke is set, the data/ready/valid are now ports of a kratos gen
        # instead of string names
        self.bespoke = bespoke

    def get_bespoke(self):
        return self.bespoke

    def get_IO_id(self):
        return self.IO_id

    def get_file_number(self):
        return self.file_number

    def get_tx_size(self):
        return self.tx_size

    def get_num_blocks(self):
        return self.num_blocks

    def get_seg_mode(self):
        return self.seg_mode

    def get_data(self):
        return self.data

    def get_valid(self):
        return self.valid

    def get_ready(self):
        return self.ready

    def get_direction(self):
        return self.direction

    def get_tensor(self):
        return self.tensor

    def get_mode(self):
        return self.mode

    def get_format(self):
        return self.format

    def connect(self, other, edge, kwargs=None):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
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
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode
        from sam.onyx.hw_nodes.pass_through_node import PassThroughNode

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
            wr_scan = other.get_name()
            new_conns = {
                'glb_to_wr_scan': [
                    ([(self.data, "io2f_17"), (wr_scan, "block_wr_in")], 17),
                ]
            }
            return new_conns
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
        elif other_type == FiberAccessNode:
            # Only could be using the write scanner portion of the fiber access
            # fa = other.get_name()
            conns_original = self.connect(other.get_write_scanner(), edge=edge)
            print(conns_original)
            conns_remapped = other.remap_conns(conns_original, "write_scanner")
            print(conns_remapped)

            return conns_remapped
        elif other_type == PassThroughNode:
            pass_through = other.get_name()
            new_conns = {
                'glb_to_pass_through': [
                    ([(self.data, "io2f_17"), (pass_through, "stream_in")], 17),
                ]
            }
            return new_conns

        else:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')

    def configure(self, attributes):
        return None
