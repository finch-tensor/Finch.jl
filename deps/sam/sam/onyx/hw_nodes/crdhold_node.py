from sam.onyx.hw_nodes.hw_node import *


class CrdHoldNode(HWNode):
    def __init__(self, name=None, outer=None, inner=None) -> None:
        super().__init__(name=name)
        self.outer = outer
        self.inner = inner

    def get_outer(self):
        return self.outer

    def get_inner(self):
        return self.inner

    def connect(self, other, edge, kwargs=None):

        crdhold = self.get_name()

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
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

        new_conns = None
        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == ReadScannerNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            conn = 0
            comment = edge.get_attributes()['comment'].strip('"')
            print("CRDHOLD TO WR SCAN")
            print(comment)
            if 'outer' in comment:
                conn = 1
            new_conns = {
                f'crdhold_{conn}_to_wr_scan': [
                    ([(crdhold, f"cmrg_coord_out_{conn}"), (wr_scan, f"data_in")], 17),
                    # ([(merge, f"cmrg_eos_out_{conn}"), (wr_scan, f"eos_in_0")], 1),
                    # ([(wr_scan, f"ready_out_0"), (merge, f"cmrg_ready_in_{conn}")], 1),
                    # ([(merge, f"cmrg_valid_out_{conn}"), (wr_scan, f"valid_in_0")], 1),
                ]
            }

            return new_conns
        elif other_type == IntersectNode:
            # out_conn = 0
            # in_conn = 0

            print(edge)
            intersect = other.get_name()
            # Use inner to process outer
            print("CRDHOLD TO INTERSECT")
            comment = edge.get_attributes()['comment'].strip('"')
            print(comment)

            label = edge.get_attributes()['label'].strip('"')
            t_label = label.split('-')[1]
            print(t_label)
            if self.get_inner() in t_label:
                out_conn = 0
            else:
                out_conn = 1

            other_t_0 = other.get_tensor_from_connection(0)
            print(other_t_0)
            if other_t_0 in comment:
                in_conn = 0
            else:
                in_conn = 1

            new_conns = {
                f'crdhold__{out_conn}_to_isect_{in_conn}': [
                    ([(crdhold, f"cmrg_coord_out_{out_conn}"), (intersect, f"coord_in_{in_conn}")], 17),
                ]
            }

        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == CrdHoldNode:
            out_conn = 1
            in_conn = 1

            other_crdhold = other.get_name()
            # Use inner to process outer
            hold_outer = other.get_outer()
            hold_inner = other.get_inner()
            conn = 0
            print(edge)
            print("CRDHOLD TO CRDHOLD")
            comment = edge.get_attributes()['comment'].strip('"')
            print(comment)
            print(hold_outer)
            print(hold_inner)
            if hold_outer in comment:
                conn = 1
            new_conns = {
                f'crdhold_to_crdhold_{conn}': [
                    ([(crdhold, f"cmrg_coord_out_{out_conn}"), (other_crdhold, f"cmrg_coord_in_{in_conn}")], 17),
                ]
            }
        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')
        elif other_type == RepSigGenNode:
            rsg = other.get_name()
            edge_comment = edge.get_attributes()['comment'].strip('"')
            if 'outer' in edge_comment:
                conn = 1
            else:
                conn = 0
            new_conns = {
                f'crdhold_to_rsg': [
                    ([(crdhold, f"cmrg_coord_out_{conn}"), (rsg, f"base_data_in")], 17),
                ]
            }
            return new_conns

        elif other_type == FiberAccessNode:
            print("CRDHOLD TO FIBER ACCESS")
            assert kwargs is not None
            assert 'flavor_that' in kwargs
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            print(kwargs)
            init_conns = self.connect(that_flavor, edge)
            print(init_conns)
            final_conns = other.remap_conns(init_conns, kwargs['flavor_that'])
            return final_conns
        else:
            raise NotImplementedError(f'Cannot connect CrdHoldNode to {other_type}')

        return new_conns

    def configure(self, attributes):
        print("CRDHOLD CONFIGURE")
        print(attributes)
        cmrg_enable = 1
        # TODO what is this supposed to be?
        cmrg_stop_lvl = 1
        op = 0
        cfg_kwargs = {
            'cmrg_enable': cmrg_enable,
            'cmrg_stop_lvl': cmrg_stop_lvl,
            'op': op
        }
        return (cmrg_enable, cmrg_stop_lvl, op), cfg_kwargs
