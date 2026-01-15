from river.drift import ADWIN

class DriftDetector:
    def __init__(self):
        self.detector = ADWIN()

    def update(self, reward):
        self.detector.update(reward)
        return self.detector.change_detected
