import gym
from typing import TypeVar
import random
import argparse

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


def getParser():
    parser = argparse.ArgumentParser(description="Random Action Wrapper")
    parser.add_argument('--epsilon', '-e',type=float, default=0.1, help="Random action probability")
    return parser.parse_args()


if __name__ == "__main__":
    args = getParser()
    env = RandomActionWrapper(gym.make("CartPole-v0"),
                              epsilon=args.epsilon)

    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break

    print(f"Reward got: {total_reward:.2f}")
