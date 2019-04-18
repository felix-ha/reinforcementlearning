import numpy as np
import random
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
plt.style.use('seaborn-paper')


class Action:
    def __init__(self, E):
        self.E = E
        self.variance = 1

    def reward(self):
        return np.random.normal(self.E, self.variance)


class Bandit:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon

        self.actions = []
        for _ in range(self.k):
            self.actions.append(Action(np.random.normal(0, 1)))

    def run(self, N):
        rewards = np.ndarray(shape=[0])
        chosen_actions = np.ndarray(shape=[0])

        for _ in range(N):
            if random.random() < self.epsilon or _ == 0:
                action = random.randint(0, self.k-1)
                reward = self.actions[action].reward()
            else:
                Q_t = np.zeros(self.k)
                for action in range(self.k):
                    indices = np.where(np.array(chosen_actions) == action)[0]
                    times_action_chosen = len(indices)
                    if times_action_chosen == 0:
                        Q_t[action] = 0
                    else:
                        Q_t[action] = np.sum(rewards[indices]) / times_action_chosen

                action = int(np.argmax(Q_t))
                reward = self.actions[action].reward()

            rewards = np.concatenate((rewards, [reward]))
            chosen_actions = np.concatenate((chosen_actions, [action]))

        return rewards


def get_avg_reward(epsilon, steps):
    k = 10
    number_simulated_bandits = 2000

    rewards_matrix = np.ndarray([0, steps])

    for i in range(number_simulated_bandits):
        bandit = Bandit(k, epsilon)
        rewards = bandit.run(steps)
        rewards_matrix = np.concatenate((rewards_matrix, [rewards]))

    return np.average(rewards_matrix, axis=0)


def compare_epsgreedy():
    steps = 1000
    t = range(steps)

    print('simulating epsilon = 0')
    reward_avg_eps0 = get_avg_reward(0, steps)

    print('simulating epsilon = 0.01')
    reward_avg_eps001 = get_avg_reward(0.01, steps)

    print('simulating epsilon = 0.1')
    reward_avg_eps01 = get_avg_reward(0.1, steps)

    print('simulating epsilon = 0.5')
    reward_avg_eps05 = get_avg_reward(0.5, steps)

    print('simulating epsilon = 1')
    reward_avg_eps1 = get_avg_reward(1, steps)

    figure(figsize=(12, 12))

    plt.plot(t, reward_avg_eps0, label='e = 0')
    plt.plot(t, reward_avg_eps001, label='e = 0.01')
    plt.plot(t, reward_avg_eps01, label='e = 0.1')
    plt.plot(t, reward_avg_eps05, label='e = 0.5')
    plt.plot(t, reward_avg_eps1, label='e = 1')

    plt.ylabel('average reward')
    plt.xlabel('steps')
    plt.legend()
    # plt.show()
    plt.savefig('kbandits.png', dpi=1000)


if __name__ == '__main__':
    compare_epsgreedy()
