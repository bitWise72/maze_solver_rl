import numpy as np
import random

class QLearningMaze:
    def __init__(self, rows, cols, walls, start, goal, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.rows = rows
        self.cols = cols
        self.walls = walls
        self.start = start
        self.goal = goal
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def get_state_key(self, state):
        return f"{state[0]},{state[1]}"

    def get_valid_actions(self, state):
        actions = []
        for i, (di, dj) in enumerate(self.actions):
            ni, nj = state[0] + di, state[1] + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols and (ni, nj) not in self.walls:
                actions.append(i)
        return actions

    def choose_action(self, state):
        key = self.get_state_key(state)
        valid_actions = self.get_valid_actions(state)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_values = self.q_table.get(key, {})
        return max(valid_actions, key=lambda a: q_values.get(a, 0.0))

    def update_q(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if next_key not in self.q_table:
            self.q_table[next_key] = {}
        max_future_q = max(self.q_table[next_key].values(), default=0.0)
        old_q = self.q_table[state_key].get(action, 0.0)
        self.q_table[state_key][action] = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)

    def train(self, episodes=1000):
        for _ in range(episodes):
            state = self.start
            while state != self.goal:
                action = self.choose_action(state)
                di, dj = self.actions[action]
                next_state = (state[0] + di, state[1] + dj)
                reward = 1 if next_state == self.goal else 0
                self.update_q(state, action, reward, next_state)
                state = next_state

    def get_policy(self):
        policy = {}
        for i in range(self.rows):
            for j in range(self.cols):
                state = (i, j)
                key = self.get_state_key(state)
                if state == self.goal or state in self.walls:
                    continue
                if key in self.q_table:
                    best_action = max(self.q_table[key], key=self.q_table[key].get)
                    policy[state] = self.actions[best_action]
        return policy
