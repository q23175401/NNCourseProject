import numpy as np
from .base_activation import Activation


class Tanh(Activation):
    def func(self, inputs):
        y = np.tanh(inputs)
        return y

    def derivative_func(self, inputs):
        return 1 - self.func(inputs)**2