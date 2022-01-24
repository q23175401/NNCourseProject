from .base_layer import Layer
import numpy as np


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)

    def Forward(self, inputs: np.ndarray):
        shape = inputs.shape
        assert (
            len(shape) == 1 + len(self.input_shape) and
            shape[1:] == self.input_shape
        ), f'input shape {shape} does not match {self.input_shape}'

        outputs = inputs.reshape([len(inputs), *self.output_shape])

        return outputs

    def Backward(self, output_gradients: np.ndarray, optimizer):
        shape = output_gradients.shape
        assert(
            len(shape) == 1 + len(self.output_shape) and
            shape[1:] == self.output_shape
        ), f'output gradients shape {shape} does not match {self.output_shape}'

        input_gradients = output_gradients.reshape(
            [len(output_gradients), *self.input_shape])

        return input_gradients
