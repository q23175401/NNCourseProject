import numpy as np
from .base_activation import Activation


class Sigmoid(Activation):
    def func(self, inputs):
        y = 1/(1+np.exp(-inputs))
        return y

    def derivative_func(self, inputs):
        delta = self.func(inputs)
        y = delta * (1-delta)
        return y
