import copy

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from random import sample
from src.agent import Agent


class DQN(torch.nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        dx = x.double()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DeepQLearningAgent(Agent):

    def __init__(self, env, lr, discount, epsilon):
        super().__init__(env)
        self.C_nn = DQN(env.observation_shape[0], env.num_actions)
        # self.C_nn = self.C_nn.double()
        self.Q_nn = DQN(env.observation_shape[0], env.num_actions)
        self.Q_nn.load_state_dict(self.C_nn.state_dict())
        # self.Q_nn = self.Q_nn.double()
        self.lr = lr
        self.optimizer = optim.AdamW(self.C_nn.parameters(), lr=lr, amsgrad=True)
        self.discount = discount
        self.epsilon = epsilon
        self.decay = 1.00001
        self.tau = 0.005
        self.l1 = torch.nn.MSELoss()
        self.memory = []
        self.steps = 0

    def best_action(self, state, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            a = np.random.choice(tuple(valid_actions), 1)[0]
        else:
            with torch.no_grad():
                out = self.Q_nn(torch.from_numpy(state))
                out_vals = np.take(out.detach().numpy(), list(valid_actions))
                a = tuple(valid_actions)[np.argmax(out_vals)]
        self._action_distribution = np.full(shape=(self._env.num_actions,), fill_value=0.5)
        self._action_distribution[a] = 1
        return a

    def train(self):
        self.Q_nn.double()
        self.C_nn.double()
        done = False
        steps = 0
        while not done:
            state, _, valid_actions, is_terminal = self._env.set_to_initial_state()
            total_reward = 0
            while not is_terminal:
                previous_state = state.copy()
                action = self.best_action(state, valid_actions)
                next_state, r, valid_actions, is_terminal = self._env.act(action)
                step = [previous_state, action, r, next_state.copy(), is_terminal]
                total_reward += r
                self.add_to_memory(step.copy())
                state = next_state
                # hash_val = hash(state.tobytes())
                # print(f"Agent did {action} and got {r}")
            steps += 1
            self.epsilon /= self.decay
            if steps % 1000 == 0:
                print(steps)
                self.steps += 1000
                # if self.train_stop():
                # if steps > 200000:
                #     done = True
                #     self.save_table()
                #     self.epsilon = -1.0
                self.copy_to_q()

            self.add_to_candidate()
            print(f"Total reward was {total_reward}.\n")

    def add_to_candidate(self):
        sampled = random.sample(self.memory, int((len(self.memory) / 2)))
        if len(sampled) < 1:
            sampled = random.sample(self.memory, 1)

        reward = torch.from_numpy(np.array([sampled[i][2] for i in range(0, len(sampled))]))
        state = torch.from_numpy(np.array([sampled[i][0] for i in range(0, len(sampled))]))
        state2 = torch.from_numpy(np.array([sampled[i][3] for i in range(0, len(sampled))]))
        action = torch.from_numpy(np.array([sampled[i][1] for i in range(0, len(sampled))]))
        terminal = torch.from_numpy(np.array([sampled[i][4] for i in range(0, len(sampled))]))

        nxt_test = self.C_nn(state)
        state_action_values = self.C_nn(state).gather(1, action.type(torch.int64).unsqueeze(1))
        next_state_values = torch.zeros(len(sampled))
        with torch.no_grad():
            next_state_values = self.Q_nn(state2).max(1)[0]
        next_state_values[terminal] = 0.0
        expected_state_action_values = (next_state_values * self.discount) + reward

        self.optimizer.zero_grad()
        crit = torch.nn.SmoothL1Loss()
        loss = crit(expected_state_action_values, state_action_values)
        # print(loss)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error
        # print("a")
        # for s in sampled:
        #     self.optimizer.zero_grad()
        #     reward = torch.from_numpy(np.array(s[3]))
        #     state = torch.from_numpy(np.array(s[0]))
        #     state2 = torch.from_numpy(np.array(s[4]))
        #     action = torch.from_numpy(np.array(s[1]))
        #     target = reward + torch.mul((self.discount * torch.max(self.C_nn(state2))), 1 - 0)
        #     current = self.C_nn(state)
        #
        #     loss = self.l1(current, target)
        #     loss.backward()  # Compute gradients
        #     self.optimizer.step()  # Backpropagate error
        #     print("a")

    def copy_to_q(self):
        self.Q_nn = copy.deepcopy(self.C_nn)
        # target_net_state_dict = self.Q_nn.state_dict()
        # policy_net_state_dict = self.C_nn.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        # self.Q_nn.load_state_dict(target_net_state_dict)

    def add_to_memory(self, entry):
        if len(self.memory) > 250:
            del self.memory[0]

        self.memory.append(entry)

    # def train_stop(self):
    #     trained = True
    #     for k in self.C_table:
    #         if k not in self.Q_table:
    #             self.Q_table[k] = np.zeros(self._env.num_actions)
    #
    #         for idx, v in enumerate(self.C_table[k]):
    #             if abs(v - self.Q_table[k][idx]) > 0.01:
    #                 trained = False
    #     return trained

    def save_table(self):
        print("saving table")

    def load_table(self):
        print("loading table")
