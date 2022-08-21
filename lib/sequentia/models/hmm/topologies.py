from .base import Topology

__all__ = ['ErgodicTopology', 'LeftRightTopology', 'LinearTopology']

class ErgodicTopology(Topology):
    pass

class LeftRightTopology(Topology):
    pass

class LinearTopology(LeftRightTopology):
    pass
