import numpy as np
from .base_loss import Loss


class CategoricalCrossEntropy(Loss):
    def Loss(self, network_y, answers):
        log_y = np.log(network_y+1e-10)
        y = -np.sum(answers * log_y, axis=1, keepdims=True)
        return y

    def Gradients(self, network_y, answers):
        x_gradients = -answers / (network_y + 1e-10)
        return x_gradients  # dE/dY
