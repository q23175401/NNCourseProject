from .base_activation import Activation
import numpy as np


class ReLU(Activation):
    def func(self, inputs):
        y = np.zeros_like(inputs)
        y[inputs >= 0] = inputs[inputs >= 0]
        return y

    def derivative_func(self, inputs):
        y = np.zeros_like(inputs)
        y[inputs >= 0] = 1
        return y
