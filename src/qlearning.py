import numpy as np
import random
from random import sample
from src.agent import Agent


class QLearningAgent(Agent):

    def __init__(self, env, lr, discount, epsilon):
        super().__init__(env)
        self.Q_table = {}
        self.C_table = {}
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.decay = 1.00001
        self.memory = []

    def best_action(self, state, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            a = np.random.choice(tuple(valid_actions), 1)[0]
        else:
            if hash(state.tobytes()) not in self.Q_table:
                self.Q_table[hash(state.tobytes())] = np.zeros(self._env.num_actions)
            # vals = [x - 1 for x in list(valid_actions)]
            valids = self.Q_table[hash(state.tobytes())][list(valid_actions)]
            a = tuple(valid_actions)[np.argmax(valids)]

        self._action_distribution = np.full(shape=(self._env.num_actions,), fill_value=0.5)
        self._action_distribution[a] = 1
        return a

    def train(self):
        done = False
        steps = 0
        while not done:
            state, _, valid_actions, is_terminal = self._env.set_to_initial_state()
            total_reward = 0
            while not is_terminal:
                action = self.best_action(state, valid_actions)
                step = [state.copy(), action, is_terminal]
                state, r, valid_actions, is_terminal = self._env.act(action)
                step.append(r)
                total_reward += r
                step.append(state.copy())
                step.append(valid_actions)
                self.add_to_memory(step.copy())
                # hash_val = hash(state.tobytes())
                # print(f"Agent did {action} and got {r}")
            steps += 1
            self.epsilon = self.epsilon / self.decay
            if steps % 1000 == 0:
                print(steps)
                if self.train_stop():
                # if steps > 5000:
                    done = True
                    self.epsilon = -1.0
                self.copy_to_q()

            self.add_to_candidate()
            # print(f"Total reward was {total_reward}.\n")
            # print(self._env.evaluate(this))
            # print(env.evaluate(agent))

    def add_to_candidate(self):
        sampled = sample(self.memory, int((len(self.memory) / 2)))
        if len(sampled) < 1:
            sampled = random.sample(self.memory, 1)

        for s in sampled:
            if len(list(s[5])) > 0:
                if not s[2]:
                    if hash(s[0].tobytes()) not in self.C_table:
                        self.C_table[hash(s[0].tobytes())] = np.zeros(self._env.num_actions)
                    if hash(s[4].tobytes()) not in self.C_table:
                        self.C_table[hash(s[4].tobytes())] = np.zeros(self._env.num_actions)
                    self.C_table[hash(s[0].tobytes())][s[1]] = (1 - self.lr) * self.C_table[hash(s[0].tobytes())][
                        s[1]] + \
                                                               self.lr * (s[3] + self.discount *
                                                                          np.max(self.C_table[hash(s[4].tobytes())][
                                                                                     list(s[5])]))
                else:
                    if hash(s[0].tobytes()) not in self.C_table:
                        self.C_table[hash(s[0].tobytes())] = np.zeros(self._env.num_actions)
                    self.C_table[hash(s[0].tobytes())][s[1]] = s[3]

    def copy_to_q(self):
        self.Q_table = self.C_table.copy()

    def add_to_memory(self, entry):
        if len(self.memory) > 100:
            del self.memory[0]

        self.memory.append(entry)

    def train_stop(self):
        trained = True
        for k in self.C_table:
            if k not in self.Q_table:
                self.Q_table[k] = np.zeros(self._env.num_actions)

            for idx, v in enumerate(self.C_table[k]):
                if abs(v - self.Q_table[k][idx]) > 0.01:
                    trained = False
        return trained


