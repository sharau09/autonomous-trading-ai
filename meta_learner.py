class MetaLearner:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def adapt(self):
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.5
