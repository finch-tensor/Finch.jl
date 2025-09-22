import enum


class ControlCodeOnyx(enum.Enum):
    STOP = 0
    DONE = 1
    MAYBE = 2


def set_bit(old_val, bit_to_set, new_bit):
    new_val = old_val | (new_bit << bit_to_set)
    return new_val


def get_bit(val, n):
    return val >> n & 1


def convert_stream_to_onyx_interp(stream):

    ctrl_op_offset = 8
    num_ctrl_bits = 2
    top_bit = 17 - 1

    converted_stream = []
    for s_ in stream:
        if type(s_) is int:
            converted_stream.append(s_)
        elif type(s_) is str:
            control_code = 0
            if 'S' in s_:
                control_code = int(s_.lstrip('S'))
            elif 'D' in s_:
                set_ctrl = ControlCodeOnyx.DONE.value
                for offset_ in range(num_ctrl_bits):
                    bts = get_bit(set_ctrl, offset_)
                    control_code = set_bit(control_code, ctrl_op_offset + offset_, bts)
            elif 'N' in s_:
                set_ctrl = ControlCodeOnyx.MAYBE.value
                for offset_ in range(num_ctrl_bits):
                    bts = get_bit(set_ctrl, offset_)
                    control_code = set_bit(control_code, ctrl_op_offset + offset_, bts)
            else:
                raise NotImplementedError
            control_code = set_bit(control_code, top_bit, 1)
            converted_stream.append(control_code)
        else:
            raise NotImplementedError
    assert len(converted_stream) == len(stream), \
        f"Input length {len(stream)} didn't match output length {len(converted_stream)}"
    return converted_stream


def print_onyx_stream(original_stream, converted_stream):
    assert len(original_stream) == len(converted_stream), f"print_onyx_stream: stream lengths don't match"
    for idx_ in range(len(original_stream)):
        print(f"{original_stream[idx_]} \t: {converted_stream[idx_]:017b}")


if __name__ == "__main__":

    example_stream = [0, 'S0', 0, 1, 2, 'S1', 'D']
    output_stream = convert_stream_to_onyx_interp(example_stream)
    print_onyx_stream(example_stream, output_stream)
