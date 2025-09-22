from sam.onyx.hw_nodes.hw_node import *


class FiberAccessNode(HWNode):
    def __init__(self, name=None,
                 read_scanner=None,
                 write_scanner=None,
                 buffet=None) -> None:
        super().__init__(name=name)
        self.read_scanner = read_scanner
        self.write_scanner = write_scanner
        self.buffet = buffet

        self.flavors = {
            'read_scanner': self.read_scanner,
            'write_scanner': self.write_scanner,
            'buffet': self.buffet
        }

    def get_read_scanner(self):
        return self.read_scanner

    def get_write_scanner(self):
        return self.write_scanner

    def get_buffet(self):
        return self.buffet

    def set_read_scanner(self, rs):
        self.flavors['read_scanner'] = rs
        self.read_scanner = rs

    def set_write_scanner(self, ws):
        self.flavors['write_scanner'] = ws
        self.write_scanner = ws

    def set_buffet(self, buf):
        self.flavors['buffet'] = buf
        self.buffet = buf

    def get_flavor(self, flavor):
        assert flavor in self.flavors
        return self.flavors[flavor]

    def remap_conns(self, conns, flavor):

        remapped_conns = {}

        for conn_set_name, conn_list in conns.items():
            # remapped_conns[conn_set_name]
            print(f"remapping {conn_set_name}: {conn_list}")
            tmp_list_conns = []
            for conn_item in conn_list:
                conns, size = conn_item
                tmp_list_actual_conns = []
                for actual_conn in conns:
                    node_name, pin_name = actual_conn
                    if node_name == self.get_name():
                        tmp_list_actual_conns.append((node_name, f"{flavor}_{pin_name}"))
                    else:
                        tmp_list_actual_conns.append((node_name, pin_name))
                tmp_list_conns.append((tmp_list_actual_conns, size))
            remapped_conns[conn_set_name] = tmp_list_conns

        return remapped_conns

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
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.stream_arbiter_node import StreamArbiterNode

        new_conns = None
        other_type = type(other)

        if other_type == GLBNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO GLB")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect FiberAccessNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect FiberAccessNode to {other_type}')
        elif other_type == ReadScannerNode:
            raise NotImplementedError(f'Cannot connect FiberAccessNode to {other_type}')
        elif other_type == WriteScannerNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO WRITE SCANNER")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == IntersectNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO INTERSECT")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == ReduceNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO Crd Hold")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect FiberAccessNode to {other_type}')
        elif other_type == MergeNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO Crd Hold")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == CrdHoldNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO Crd Hold")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == RepeatNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO Crd Hold")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == ComputeNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO WRITE SCANNER")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect FiberAccessNode to {other_type}')
        elif other_type == RepSigGenNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO Crd Hold")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns

        elif other_type == FiberAccessNode:

            assert kwargs is not None
            assert 'flavor_this' in kwargs
            assert 'flavor_that' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            print(kwargs)
            print("FIBER ACCESS TO FIBER ACCESS")
            init_conns = this_flavor.connect(that_flavor, edge)
            print(init_conns)
            final_conns_1 = self.remap_conns(init_conns, kwargs['flavor_this'])
            final_conns_2 = other.remap_conns(final_conns_1, kwargs['flavor_that'])
            print(final_conns_2)
            return final_conns_2
        elif other_type == StreamArbiterNode:
            assert kwargs is not None
            assert 'flavor_this' in kwargs
            this_flavor = self.get_flavor(kwargs['flavor_this'])
            print(kwargs)
            print("FIBER ACCESS TO Stream Arbiter")
            init_conns = this_flavor.connect(other, edge)
            print(init_conns)
            final_conns = self.remap_conns(init_conns, kwargs['flavor_this'])
            return final_conns
        else:
            raise NotImplementedError(f'Cannot connect FiberAccessNode to {other_type}')

        return new_conns

    def configure(self, attributes, flavor):

        cfg_tuple, cfg_kwargs = self.get_flavor(flavor=flavor).configure(attributes)
        cfg_kwargs['flavor'] = flavor
        print("THESE ARE MY CONFIG KWARGS")
        print(cfg_kwargs)
        # breakpoint()

        # vr_mode = 0
        # cfg_tuple += (vr_mode,)
        # cfg_kwargs["vr_mode"] = vr_mode

        return cfg_tuple, cfg_kwargs
