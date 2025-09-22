from sam.onyx.hw_nodes.hw_node import *


class RepeatNode(HWNode):
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
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

        repeat = self.get_name()
        new_conns = None
        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == ReadScannerNode:
            rd_scan = other.get_name()
            new_conns = {
                f'repeat_to_read_scan': [
                    # send output to rd scanner
                    ([(repeat, f"ref_data_out"), (rd_scan, f"us_pos_in")], 17),
                    # ([(repeat, f"ref_eos_out"), (rd_scan, f"us_eos_in")], 1),
                    # ([(rd_scan, f"us_ready_out"), (repeat, f"ref_ready_in")], 1),
                    # ([(repeat, f"ref_valid_out"), (rd_scan, f"us_valid_in")], 1),
                ]
            }

            return new_conns
        elif other_type == WriteScannerNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == IntersectNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == RepeatNode:
            other_repeat = other.get_name()
            new_conns = {
                f'repeat_to_repeat': [
                    # send output to rd scanner
                    ([(repeat, f"ref_data_out"), (other_repeat, "proc_data_in")], 17),
                ]
            }

            return new_conns
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')
        elif other_type == CrdHoldNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        elif other_type == FiberAccessNode:
            print("REPEAT TO FIBER ACCESS")
            assert kwargs is not None
            assert 'flavor_that' in kwargs
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            print(kwargs)
            init_conns = self.connect(that_flavor, edge)
            print(init_conns)
            final_conns = other.remap_conns(init_conns, kwargs['flavor_that'])
            return final_conns
        else:
            raise NotImplementedError(f'Cannot connect RepeatNode to {other_type}')

        return new_conns

    def configure(self, attributes):

        spacc_mode = 0
        if 'spacc' in attributes:
            spacc_mode = 1

        print("Repeat stop")
        root = 0
        stop_lvl = 1
        if 'true' in attributes['root'].strip('"'):
            root = 1
            stop_lvl = 0
        print(attributes)
        cfg_kwargs = {
            'stop_lvl': stop_lvl,
            'root': root,
            'spacc_mode': spacc_mode
        }
        return (stop_lvl, root, spacc_mode), cfg_kwargs
