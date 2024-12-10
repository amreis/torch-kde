from abc import ABC, abstractmethod

SUPPORTED_ALGORITHMS = [
    "standard"
    #TODO: Add algorithms like KDTree and BallTree
]

class Tree(ABC):
    def __init__(self):
        self.data = None

    @abstractmethod
    def build(self, X):
        pass


class RootTree(Tree):
    """Standard algorithm for kernel density estimation. 
       This algorithm simply builds a root node that returns all the data points."""
    def build(self, X):
        self.data = X
        return self

    def query(self, x, return_distance=False):
        assert return_distance == False, "Distance computation is not supported."
        # TODO: Implement return_distance
        return self.data
    