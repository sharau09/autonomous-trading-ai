import torch
import torch.optim as optim
from agent.policy import PolicyNetwork
from agent.drift_detector import DriftDetector
from agent.meta_learner import MetaLearner

class TradingAgent:
    def __init__(self):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.drift = DriftDetector()
        self.meta = MetaLearner(self.optimizer)

    def act(self, state):
        state = torch.FloatTensor(state)
        return torch.argmax(self.policy(state)).item()

    def learn(self, reward):
        if self.drift.update(reward):
            self.meta.adapt()
            return True
        return False
