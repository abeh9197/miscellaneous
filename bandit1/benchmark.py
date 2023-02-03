from typing import List
from ThompsonSampling import thompson_sampling
from Arm import Arm
import numpy as np


def simulate_thompson_sampling():
    arms: List[Arm] = [Arm(0.3) for i in range(4)]
    arms.append(Arm(0.5))
    return thompson_sampling(arms=arms, T=10**3)


def __out_benchmark(array):
    print(f"\t Avg: {np.average(array)}, Std:{np.std(array)}, Max: {np.max(array)}, Min:{np.min(array)}")


if __name__ == "__main__":
    loop_count = 100
    thompson_sampling_reward_hist = []
    for i in range(loop_count):
        thompson_sampling_reward_hist.append(simulate_thompson_sampling())
        print("Thompson sampling")
        __out_benchmark(thompson_sampling_reward_hist)