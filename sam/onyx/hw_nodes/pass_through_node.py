from sam.onyx.hw_nodes.hw_node import *


class PassThroughNode(HWNode):
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
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

        new_conns = None
        pass_through = self.get_name()

        other_type = type(other)

        if other_type == WriteScannerNode:
            wr_scan = other.get_name()
            new_conns = {
                'pass_through_to_wr_scan': [
                    ([(pass_through, "stream_out"), (wr_scan, "block_wr_in")], 17),
                ]
            }
            return new_conns
        elif other_type == FiberAccessNode:
            # Only could be using the write scanner portion of the fiber access
            # fa = other.get_name()
            conns_original = self.connect(other.get_write_scanner(), edge=edge)
            print(conns_original)
            conns_remapped = other.remap_conns(conns_original, "write_scanner")
            print(conns_remapped)

            return conns_remapped

        else:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')

        return new_conns

    def update_input_connections(self):
        self.num_inputs_connected += 1

    def get_num_inputs(self):
        return self.num_inputs_connected

    def configure(self, attributes):
        # print("Pass Through CONFIGURE")
        # print(attributes)

        placeholder = 1
        cfg_kwargs = {
            'placeholder': placeholder
        }
        return (placeholder), cfg_kwargs
