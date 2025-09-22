from sam.onyx.hw_nodes.hw_node import *


class StreamArbiterNode(HWNode):
    def __init__(self, name=None) -> None:
        super().__init__(name=name)
        self.max_num_inputs = 4
        self.num_inputs_connected = 0
        self.num_outputs = 1
        self.num_outputs_connected = 0

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
        stream_arb = self.get_name()

        other_type = type(other)

        if other_type == GLBNode:
            other_data = other.get_data()
            other_ready = other.get_ready()
            other_valid = other.get_valid()
            new_conns = {
                'stream_arbiter_to_glb': [
                    ([(stream_arb, "stream_out"), (other_data, "f2io_17")], 17),
                ]
            }
            return new_conns
        elif other_type == StreamArbiterNode:
            cur_inputs = other.get_num_inputs()
            assert cur_inputs < self.max_num_inputs - 1, f"Cannot connect StreamArbiterNode to {other_type}, too many inputs"
            down_stream_arb = other.get_name()
            new_conns = {
                f'stream_arbiter_to_stream_arbiter_{cur_inputs}': [
                    ([(stream_arb, "stream_out"), (down_stream_arb, f"stream_in_{cur_inputs}")], 17),
                ]
            }
            other.update_input_connections()
            return new_conns
        else:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')

        return new_conns

    def update_input_connections(self):
        self.num_inputs_connected += 1

    def get_num_inputs(self):
        return self.num_inputs_connected

    def configure(self, attributes):
        # print("STREAM ARBITER CONFIGURE")
        # print(attributes)

        seg_mode = attributes['seg_mode']
        num_requests = self.num_inputs_connected
        assert num_requests > 0, "StreamArbiterNode must have at least one input"
        num_requests = num_requests - 1  # remap to the range of 0-3

        cfg_kwargs = {
            'num_requests': num_requests,
            'seg_mode': seg_mode
        }
        return (num_requests, seg_mode), cfg_kwargs
