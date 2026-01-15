import numpy as np
import pandas as pd

class TradingEnv:
    def __init__(self):
        self.data = pd.read_csv("data/market_data.csv")
        self.reset()

    def reset(self):
        self.step_index = 0
        self.balance = 10000.0
        self.shares = 0
        self.prev_value = self.balance
        return self._get_state()

    def _get_state(self):
        idx = min(self.step_index, len(self.data) - 1)
        price = self.data.iloc[idx]["price"]
        return np.array([price, self.balance, self.shares], dtype=np.float32)

    def step(self, action):
        # Stop if data finished
        if self.step_index >= len(self.data) - 1:
            return self._get_state(), 0.0, True

        price = self.data.iloc[self.step_index]["price"]

        # 0 = Buy, 1 = Sell, 2 = Hold
        if action == 0 and self.balance >= price:
            self.balance -= price
            self.shares += 1

        elif action == 1 and self.shares > 0:
            self.balance += price
            self.shares -= 1

        self.step_index += 1

        next_price = self.data.iloc[self.step_index]["price"]
        total_value = self.balance + self.shares * next_price
        reward = total_value - self.prev_value
        self.prev_value = total_value

        return self._get_state(), reward, False
