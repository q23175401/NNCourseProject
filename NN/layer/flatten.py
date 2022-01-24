from .base_layer import Layer
from .reshape import Reshape
import numpy as np


class Flatten(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        items = 1
        for s in input_shape:
            items *= s
        self.output_dim = items
        self.Reshape = Reshape(input_shape, [items])

    def Forward(self, inputs: np.ndarray):
        return self.Reshape.Forward(inputs)

    def Backward(self, outputgradients: np.ndarray, optimizer):
        return self.Reshape.Backward(outputgradients, optimizer)
