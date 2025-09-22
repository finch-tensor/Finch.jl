import numpy
from sam.onyx.generate_matrices import *


class FiberTreeFiber():

    def __init__(self, parent=None):

        self.parent = parent
        self.coords_payloads = []
        # self.payloads = []

    def add_coord_payload_tuple(self, coord_payload_tuple):
        # crd, payload = coord_payload_tuple
        # self.coords.append(crd)
        # self.payloads.append(payload)
        self.coords_payloads.append(coord_payload_tuple)

    def get_coord_payloads(self):
        return self.coords_payloads

    def remove_coord_payload_tuple(self, coord_payload_tuple):
        self.coords_payloads.remove(coord_payload_tuple)


class FiberTree():
    def __init__(self, tensor=None) -> None:
        assert tensor is not None
        self.tensor = tensor

        self.root_fiber = FiberTreeFiber(parent=None)

        sub_tensor = self.tensor[:]
        self.populate_fiber(self.root_fiber, sub_tensor=sub_tensor)
        self.clean_fiber_tree()

    def get_root(self):
        return self.root_fiber

    def populate_fiber(self, fiber, sub_tensor):
        # Last level detection
        if len(sub_tensor.shape) == 1:
            # Finally have just a row, this is the base case...(could be a scalar)
            for crd, sub_sub_tensor in enumerate(sub_tensor):
                # This is vals...(can check type of payloads)
                fiber.add_coord_payload_tuple((crd, sub_sub_tensor))
                # self.populate_fiber(tmp_fiber, sub_sub_tensor)
        else:
            for crd, sub_sub_tensor in enumerate(sub_tensor):
                tmp_fiber = FiberTreeFiber(parent=fiber)
                fiber.add_coord_payload_tuple((crd, tmp_fiber))
                self.populate_fiber(tmp_fiber, sub_sub_tensor)

    def clean_fiber_tree(self):
        '''
        This routine will clean the fiber by getting rid of 0 valued terminals
        '''
        self._clean_fiber_tree_helper(self.root_fiber)

    def _clean_fiber_tree_helper(self, fiber):

        self.clean_marks = []
        for crd, payload in fiber.get_coord_payloads().copy():
            # Base case...
            if type(payload) is not FiberTreeFiber:
                # If it's a 0, we want to clear it from the original list
                if payload == 0:
                    fiber.remove_coord_payload_tuple((crd, payload))
                else:
                    pass
            else:
                # If there are children payloads, then we need to recursively clean them and
                # then remove them if they're empty
                self._clean_fiber_tree_helper(payload)
                if len(payload.get_coord_payloads()) == 0:
                    fiber.remove_coord_payload_tuple((crd, payload))

    def __str__(self):
        base_str = ""
        for crd, payload in self.root_fiber.get_coord_payloads():
            base_str += f"{crd}\t"
        base_str += "\n"
        for crd, payload in self.root_fiber.get_coord_payloads():
            base_str += "["
            for crd2, payload2 in payload.get_coord_payloads():
                base_str += f"{crd2}\t"
            base_str += "]"
        return base_str


if __name__ == "__main__":
    random.seed(10)
    numpy.random.seed(10)
    mg = MatrixGenerator(name='B', shape=[10, 10], dump_dir='/home/max/Documents/SPARSE/sam/OUTPUTS_DUMP', sparsity=0.8)
    array = mg.get_matrix()
    print(array)
    ft = FiberTree(tensor=array)
    print(ft)
