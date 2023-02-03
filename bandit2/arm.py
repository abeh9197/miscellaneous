"""Arm"""


from numpy.random import binomial


class Arm:
    """Arm class for common use."""
    def __init__(self, probability: float) -> None:
        self.__probability = probability
        self.success = 0
        self.fail = 0

    def play(self) -> int:
        """Methods meant to pull the arm."""
        result = binomial(n=1, p=self.__probability)
        if result == 1:
            self.success += 1
        else:
            self.fail += 1
        return result