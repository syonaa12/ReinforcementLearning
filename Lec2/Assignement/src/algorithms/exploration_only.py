import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class ExplorationOnly(BaseMABAlgorithm):
    """
    Pure exploration algorithm - randomly selects arms
    """
    def __init__(self, n_arms: int, **kwargs):
        super().__init__(n_arms, **kwargs)
        
    def select_arm(self) -> int:
        """
        Selects an arm randomly with equal probability.

        Returns:
            int: Randomly selected arm index between 0 and self.n_arms - 1
        """
        return np.random.randint(0, self.n_arms)
