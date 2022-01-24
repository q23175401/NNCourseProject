import numpy as np
from .base_loss import Loss


class LeastSquaredError(Loss):
    def Loss(self, network_y, answers):
        err = 0.5 * np.power(network_y - answers, 2)
        return err

    def Gradients(self, network_y, answers):
        return network_y - answers
