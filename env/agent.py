import pickle
import os
import random
from collections import defaultdict

from config import ALPHA, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY

# actions paddle can take
ACTION_UP   = 0
ACTION_STAY = 1
ACTION_DOWN = 2
ACTIONS = [ACTION_UP, ACTION_STAY, ACTION_DOWN]


class QLearningAgent:
    def __init__(self, name="agent", epsilon=EPSILON_START):
        self.name = name
        self.epsilon = epsilon
        # default q values are 0 for all actions
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])

        self.total_rallies = 0
        self.total_steps = 0
        self.wins = 0
        self.losses = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q = self.q_table[state]
        return q.index(max(q))

    def learn(self, state, action, reward, next_state, done):
        q_cur = self.q_table[state][action]
        if done:
            q_next_max = 0.0
        else:
            q_next_max = max(self.q_table[next_state])
        td_error = (reward + GAMMA * q_next_max) - q_cur
        self.q_table[state][action] += ALPHA * td_error
        self.total_steps += 1

    def on_rally_end(self):
        # called after each rally ends, decays epsilon over time
        self.total_rallies += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "total_rallies": self.total_rallies,
            "total_steps": self.total_steps,
            "wins": self.wins,
            "losses": self.losses,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[{self.name}] saved to {path} (rallies={self.total_rallies}, epsilon={self.epsilon:.4f})")

    def load(self, path):
        if not os.path.exists(path):
            print(f"[{self.name}] no saved model found at {path}, starting fresh")
            return
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.q_table.update(d["q_table"])
        self.epsilon = d.get("epsilon", EPSILON_MIN)
        self.total_rallies = d.get("total_rallies", 0)
        self.total_steps = d.get("total_steps", 0)
        self.wins = d.get("wins", 0)
        self.losses = d.get("losses", 0)
        print(f"[{self.name}] loaded from {path} (rallies={self.total_rallies}, epsilon={self.epsilon:.4f}, states={len(self.q_table)})")

    @property
    def q_table_size(self):
        return len(self.q_table)