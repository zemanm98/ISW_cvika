import copy

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from random import sample
from src.agent import Agent
import wandb

class DQN(torch.nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 8)
        # self.layer2 = torch.nn.Linear(16, 8)
        self.layer3 = torch.nn.Linear(8, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # dx = x.double()
        x = F.sigmoid(self.layer1(x))
        # x = F.sigmoid(self.layer2(x))
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
        self.tau = 0.05
        self.l1 = torch.nn.MSELoss()
        self.memory = []
        self.steps = 0

    def best_action(self, state, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            a = np.random.choice(tuple(valid_actions), 1)[0]
        else:
            with torch.no_grad():
                out = self.Q_nn(torch.from_numpy(state).float())
                out_vals = np.take(out.detach().numpy(), list(valid_actions))

                # filtrovat do budoucna pres valid actions
                a = tuple(valid_actions)[np.argmax(out_vals)]
        self._action_distribution = np.full(shape=(self._env.num_actions,), fill_value=0.5)
        self._action_distribution[a] = 1
        return a

    def train(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="isw_neuronka"
        )
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
                state = next_state.copy()
            steps += 1
            self.add_to_candidate()
            self.epsilon /= self.decay
            if self.epsilon <= 0.2:
                self.decay = 1.0
            if steps % 2 == 0 and steps != 0:
                self.steps += 2
                # if self.train_stop():
                self.copy_to_q()
            wandb.log({"reward": total_reward})

    def add_to_candidate(self):
        if len(self.memory) > 129:
            sampled = random.sample(self.memory, 128)

            reward = torch.from_numpy(np.array([sampled[i][2] for i in range(0, len(sampled))]))
            state = torch.from_numpy(np.array([sampled[i][0] for i in range(0, len(sampled))]))
            state2 = torch.from_numpy(np.array([sampled[i][3] for i in range(0, len(sampled))]))
            action = torch.from_numpy(np.array([sampled[i][1] for i in range(0, len(sampled))]))
            terminal = torch.from_numpy(np.array([sampled[i][4] for i in range(0, len(sampled))]))

            state_action_values = self.C_nn(state.float()).gather(1, action.type(torch.int64).unsqueeze(1))

            next_state_values = torch.zeros(len(sampled))
            with torch.no_grad():
                next_state_values = self.Q_nn(state2.float()).max(1)[0]

            next_state_values[terminal] = 0.0
            expected_state_action_values = (next_state_values * self.discount) + reward

            crit = torch.nn.MSELoss()
            loss = crit(state_action_values, expected_state_action_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.C_nn.parameters(), 5)
            self.optimizer.step()
            wandb.log({"loss": loss})

    def copy_to_q(self):
        self.Q_nn.load_state_dict(self.C_nn.state_dict())

    def add_to_memory(self, entry):
        if len(self.memory) > 1000:
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
