from numpy import block
from sam.onyx.hw_nodes.hw_node import *


class ReadScannerNode(HWNode):
    def __init__(self, name=None, tensor=None,
                 mode=None, dim_size=None, index=None,
                 format=None) -> None:
        super().__init__(name=name)
        self.tensor = tensor
        self.mode = mode
        self.dim_size = dim_size
        self.index = index
        self.format = format

    def get_tensor(self):
        return self.tensor

    def get_mode(self):
        return self.mode

    def get_dim_size(self):
        return self.dim_size

    def get_index(self):
        return self.index

    def get_format(self):
        return self.format

    def connect(self, other, edge, kwargs=None):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.stream_arbiter_node import StreamArbiterNode

        new_conns = None
        rd_scan = self.get_name()
        other_type = type(other)

        if other_type == GLBNode:
            other_data = other.get_data()
            other_ready = other.get_ready()
            other_valid = other.get_valid()
            new_conns = {
                'rd_scan_to_glb': [
                    ([(rd_scan, "block_rd_out"), (other_data, "f2io_17")], 17),
                ]
            }
            return new_conns
        elif other_type == BuffetNode:
            buffet = other.get_name()
            new_conns = {
                'buffet_to_rd_scan': [
                    # rd rsp
                    ([(buffet, "rd_rsp_data"), (rd_scan, "rd_rsp_data_in")], 17),
                    # addr
                    ([(rd_scan, "addr_out"), (buffet, "rd_addr")], 17),
                    # op
                    ([(rd_scan, "op_out"), (buffet, "rd_op")], 17),
                    # id
                    ([(rd_scan, "ID_out"), (buffet, "rd_ID")], 17),
                ]
            }
            return new_conns
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == ReadScannerNode:
            # send the ref to the next rd scanner
            other_rd_scan = other.get_name()
            new_conns = {
                'rd_scan_to_rd_scan': [
                    ([(rd_scan, "pos_out"), (other_rd_scan, "us_pos_in")], 17),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            # send the crd to write scanner
            wr_scan = other.get_name()
            edge_attr = edge.get_attributes()
            if 'use_alt_out_port' in edge_attr:
                out_conn = 'block_rd_out'
            elif ('vector_reduce_mode' in edge_attr):
                if (edge_attr['vector_reduce_mode']):
                    out_conn = 'pos_out'
            else:
                out_conn = 'coord_out'

            new_conns = {
                'rd_scan_to_wr_scan': [
                    ([(rd_scan, out_conn), (wr_scan, "data_in")], 17),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            # Send both....
            isect = other.get_name()
            if 'vector_reduce_mode' in edge.get_attributes():
                if edge.get_attributes()['vector_reduce_mode']:
                    isect_conn = 1
            elif 'special' in edge.get_attributes():
                isect_conn = 0
            else:
                isect_conn = other.get_connection_from_tensor(self.get_tensor())

            e_attr = edge.get_attributes()
            # isect_conn = 0
            # if self.get_tensor() == 'C' or self.get_tensor() == 'c':
            #     isect_conn = 1
            e_type = e_attr['type'].strip('"')
            if "crd" in e_type:
                new_conns = {
                    f'rd_scan_to_isect_{isect_conn}_crd': [
                        # send output to rd scanner
                        ([(rd_scan, "coord_out"), (isect, f"coord_in_{isect_conn}")], 17),
                    ]
                }
            elif 'ref' in e_type:

                rd_scan_out_port = "pos_out"

                if 'val' in e_attr and e_attr['val'].strip('"') == 'true':
                    rd_scan_out_port = "coord_out"

                new_conns = {
                    f'rd_scan_to_isect_{isect_conn}_pos': [
                        # send output to rd scanner
                        ([(rd_scan, rd_scan_out_port), (isect, f"pos_in_{isect_conn}")], 17),
                    ]
                }
            else:
                raise NotImplementedError(f'Only accept ref or crd types to intersect....you used {type}')
            # other.update_input_connections()
            return new_conns
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == MergeNode:

            edge_attr = edge.get_attributes()
            crddrop = other.get_name()
            print("CHECKING READ TENSOR - CRDDROP")
            print(edge)
            print(self.get_tensor())
            crd_drop_outer = other.get_outer()
            comment = edge_attr['comment'].strip('"')
            conn = 0
            # okay this is dumb, stopgap until we can have super consistent output
            try:
                mapped_to_conn = comment.split("-")[1]
            except Exception:
                try:
                    mapped_to_conn = comment.split("_")[1]
                except Exception:
                    mapped_to_conn = comment
            if crd_drop_outer in mapped_to_conn:
                conn = 1

            if 'use_alt_out_port' in edge_attr:
                out_conn = 'block_rd_out'
            elif ('vector_reduce_mode' in edge_attr):
                if (edge_attr['vector_reduce_mode']):
                    out_conn = 'pos_out'
            else:
                out_conn = 'coord_out'

            new_conns = {
                f'rd_scan_to_crddrop_{conn}': [
                    ([(rd_scan, out_conn), (crddrop, f"coord_in_{conn}")], 17),
                ]
            }

            return new_conns

        elif other_type == RepeatNode:
            repeat = other.get_name()
            new_conns = {
                'rd_scan_to_repeat': [
                    # send output to rd scanner
                    ([(rd_scan, "pos_out"), (repeat, "proc_data_in")], 17),
                ]
            }
            return new_conns
        elif other_type == ComputeNode:
            compute = other.get_name()
            # Can use dynamic information to assign inputs to compute nodes
            # since add/mul are commutative
            compute_conn = other.get_num_inputs()
            edge_attr = edge.get_attributes()
            if "specified_port" in edge_attr and edge_attr["specified_port"] is not None:
                compute_conn = edge_attr["specified_port"]
                other.mapped_input_ports.append(compute_conn.strip("data"))
                new_conns = {
                    f'rd_scan_to_compute_{compute_conn}': [
                        ([(rd_scan, "coord_out"), (compute, f"{compute_conn}")], 17),
                    ]
                }
            else:
                compute_conn = other.mapped_input_ports[compute_conn]
                new_conns = {
                    f'rd_scan_to_compute_{compute_conn}': [
                        ([(rd_scan, "coord_out"), (compute, f"data{compute_conn}")], 17),
                    ]
                }
                # Now update the PE/compute to use the next connection next time
                other.update_input_connections()

            return new_conns

        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == RepSigGenNode:
            rsg = other.get_name()
            edge_attr = edge.get_attributes()
            if 'vr_special' in edge_attr:
                new_conns = {
                    f'rd_scan_to_rsg': [
                        ([(rd_scan, "pos_out"), (rsg, f"base_data_in")], 17),
                    ]
                }
            else:
                new_conns = {
                    f'rd_scan_to_rsg': [
                        ([(rd_scan, "coord_out"), (rsg, f"base_data_in")], 17),
                    ]
                }
            return new_conns
        elif other_type == CrdHoldNode:
            crdhold = other.get_name()
            # Use inner to process outer
            crdhold_outer = other.get_outer()
            crdhold_inner = other.get_inner()
            conn = 0
            print(edge)
            print("RDSCAN TO CRDHOLD")
            comment = edge.get_attributes()['comment'].strip('"')
            print(comment)
            mapped_to_conn = comment
            if crdhold_outer in mapped_to_conn:
                conn = 1
            new_conns = {
                f'rd_scan_to_crdhold_{conn}': [
                    ([(rd_scan, "coord_out"), (crdhold, f"cmrg_coord_in_{conn}")], 17),
                ]
            }

            return new_conns
        elif other_type == StreamArbiterNode:
            cur_inputs = other.get_num_inputs()
            assert cur_inputs <= other.max_num_inputs - 1, f"Cannot connect ReadScannerNode to {other_type}, too many inputs"
            down_stream_arb = other.get_name()
            new_conns = {
                f'rd_scan_to_stream_arbiter_{cur_inputs}': [
                    ([(rd_scan, "block_rd_out"), (down_stream_arb, f"stream_in_{cur_inputs}")], 17),
                ]
            }
            other.update_input_connections()
            return new_conns
        else:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')

        return new_conns

    def configure(self, attributes):
        inner_offset = 0
        max_outer_dim = 0
        strides = [0]
        ranges = [1]
        dense = 0
        dim_size = 1
        stop_lvl = 0

        # if 'spacc' in attributes:
        #    spacc_mode = 1
        #    assert 'stop_lvl' in attributes
        #    stop_lvl = int(attributes['stop_lvl'].strip('"'))
        # else:
        #    spacc_mode = 0

        # This is a fiberwrite's opposing read scanner for comms with GLB
        if attributes['type'].strip('"') == 'fiberwrite':
            # in fiberwrite case, we are in block mode
            mode = attributes['mode'].strip('"')
            if mode == 'vals' or int(mode) != 0:
                is_root = 0
            else:
                is_root = 1
        elif attributes['type'].strip('"') == 'arrayvals':
            is_root = 0
        else:
            is_root = int(attributes['root'].strip('"') == 'true')
            if attributes['format'].strip('"') == 'dense':
                print("FOUND DENSE")
                dense = 1
                dim_size = self.dim_size
        do_repeat = 0
        repeat_outer = 0
        repeat_factor = 0
        if attributes['type'].strip('"') == 'arrayvals':
            # stop_lvl = 0
            lookup = 1
        elif attributes['mode'].strip('"') == 'vals':
            # stop_lvl = 0
            lookup = 1
        else:
            lookup = 0
        block_mode = int(attributes['type'].strip('"') == 'fiberwrite')

        if 'vector_reduce_mode' in attributes:
            is_in_vr_mode = attributes['vector_reduce_mode'].strip('"')
            if is_in_vr_mode == "true":
                vr_mode = 1
        else:
            vr_mode = 0

        glb_addr_base = 0
        glb_addr_stride = 0
        if 'glb_addr_base' in attributes:
            glb_addr_base = int(attributes['glb_addr_base'])
        if 'glb_addr_stride' in attributes:
            glb_addr_stride = int(attributes['glb_addr_stride'])

        cfg_kwargs = {
            'dense': dense,
            'dim_size': dim_size,
            'inner_offset': inner_offset,
            'max_out': max_outer_dim,
            'strides': strides,
            'ranges': ranges,
            'root': is_root,
            'do_repeat': do_repeat,
            'repeat_outer': repeat_outer,
            'repeat_factor': repeat_factor,
            # 'stop_lvl': stop_lvl,
            'block_mode': block_mode,
            'lookup': lookup,
            # 'spacc_mode': spacc_mode
            'vr_mode': vr_mode,
            'glb_addr_base': glb_addr_base,
            'glb_addr_stride': glb_addr_stride
        }

        return (inner_offset, max_outer_dim, strides, ranges, is_root, do_repeat,
                repeat_outer, repeat_factor, block_mode, lookup, vr_mode), cfg_kwargs
