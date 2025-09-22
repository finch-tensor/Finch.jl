# NOT CURRENTLY USED FILE


class FiberCoordinateLookupError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Fiber():

    def __init__(self,
                 coordinates=[],
                 payloads=[],
                 format="CSR"):

        self.coordinates = coordinates
        self.payloads = payloads

        self.cpmap = {}

        for (idx, coord) in enumerate(self.coordinates):
            self.cpmap[coord] = self.payloads[idx]

    def get_coordinates(self):
        return self.coordinates

    def get_payloads(self):
        return self.payloads

    def lookup_payload(self, coord):

        if coord not in self.cpmap:
            raise FiberCoordinateLookupError
        else:
            return self.cpmap[coord]
