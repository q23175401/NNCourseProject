import numpy as np
from abc import ABC, abstractmethod
from ..base_layer import Layer


class Activation(Layer, ABC):
    def __init__(self):
        self.last_inputs = None

    @abstractmethod
    def func(self, inputs):
        raise NotImplementedError(
            'You need to implement func of this activation function')

    @abstractmethod
    def derivative_func(self, inputs):
        raise NotImplementedError(
            'You need to implement derivative func of this activation function')

    def Forward(self, inputs):
        self.last_inputs = inputs
        y = self.func(inputs)
        return y

    def Backward(self, output_gradients, _optimizer):
        assert self.last_inputs is not None, "you need to forward before backward"

        x_gradients = self.derivative_func(self.last_inputs)
        return output_gradients * x_gradients
