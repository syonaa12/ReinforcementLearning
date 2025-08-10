import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class UCB(BaseMABAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm.
    Balances exploration and exploitation using confidence bounds.
    """
    def __init__(self, n_arms: int, c: float = 2.0, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.c = c  # Exploration parameter

    def select_arm(self) -> int:
        """
        Selects an arm using the UCB algorithm.

        Returns:
            int: Index of the selected arm.
        """
        # 1. Check for unpulled arms
        unpulled_arms = np.where(self.pulls == 0)[0]
        if len(unpulled_arms) > 0:
            return int(unpulled_arms[0])  # Pull untried arm first

        # 2. Total number of pulls so far
        total_pulls = np.sum(self.pulls)

        # 3. Compute UCB values
        confidence_bounds = self.c * np.sqrt(np.log(total_pulls) / self.pulls)
        ucb_values = self.estimates + confidence_bounds

        # 4. Return the arm with the highest UCB value
        return int(np.argmax(ucb_values))
