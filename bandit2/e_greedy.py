"""
ハイパーパラメータϵに適当な値（0.3とか0.8とか）をセットする。
確率ϵで表のでるコインを1回投げる。
表が出た場合、探索を行う。すなわち、全アームから一様ランダムにアームを選択し、選択したームを引く。
裏が出た場合、活用を行う。すなわち、今までで最も成功割合の高いアームを選択し、選択したアームを引く。
2～5をT回繰り返す
"""


from numpy.random import binomial, randint
from arm import Arm


def calc_success_ratio(arm: Arm) -> float:
    """Returns ratio of success."""
    if arm.success + arm.fail == 0:
        return 0
    return arm.success / arm.success + arm.fail

def epsilon_greedy(arms, t: int, epsilon) -> int:
    """
    'arms' is list of Arm,
    't' is for time.
    """
    reward = 0
    for i in range(t):
        if binomial(n=1, p=epsilon) == 1:
            # NOTE: Choose the arm with random.(Exploration)
            index = randint(0, 99)
        else:
            # NOTE: Choose the arm with the best chance of success so far. (Exploitation)
            averages = calc_success_ratio(arm=)