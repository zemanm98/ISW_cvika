from src.agent import RandomAgent, BalancingAgent
from src.environment import Connect4, TicTacToe, FrozenLake, CartPole, MountainCar, LectureExample, MultiArmedBandit
from src.play import Simulator
from src.qlearning import QLearningAgent
from src.deep_qlearning import DeepQLearningAgent
from collections import Counter

if __name__ == '__main__':
    # env = MultiArmedBandit()
    # env = FrozenLake()
    env = CartPole()
    agent = DeepQLearningAgent(env, 0.0001, 0.99, 0.99)
    agent.train()

    # print(env.evaluate(agent))
    # env = MultiArmedBandit()
    # env = FrozenLake()
    # env = LectureExample()
    # agent = RandomAgent(env)
    # state, _, valid_actions, is_terminal = env.set_to_initial_state()
    # total_reward = 0
    # while not is_terminal:
    #     random_action = agent.best_action(state, valid_actions)
    #     state, r, valid_actions, is_terminal = env.act(random_action)
    #     total_reward += r
    #     print(f"Agent did {random_action} and got {r}")
    # print(f"Total reward was {total_reward}.")
    print(env.evaluate(agent))
    # print(env.evaluate(agent))
    #env = CartPole()
    #agent = RandomAgent(env)
    #sim = Simulator(env, agent, fps=15)
    # sim = Simulator(env, agent, fps=0)
    #sim.show()
    #sim.run()
    # sim = Simulator(env, agent, fps=15)
    sim = Simulator(env, agent, fps=0)
    sim.show()
    sim.run()
