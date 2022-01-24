from .base_loss import Loss
import numpy as np


class MeanSquaredError(Loss):
    def Loss(self, network_y, answers):
        err = np.power(network_y - answers, 2)
        return err.mean(axis=1, keepdims=True)

    def Gradients(self, network_y, answers):
        return (network_y - answers)*2/(network_y.shape[1]+1e-10)
