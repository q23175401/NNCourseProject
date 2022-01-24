import numpy as np
from .base_layer import Layer
from ..optimizer import Optimizer


class DenseLayer(Layer):
    def __init__(self, n_inputs,  n_units):
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.create_trainable_parameters()

    def create_trainable_parameters(self):
        self.W = np.random.uniform(-1, 1, size=[self.n_inputs, self.n_units])
        self.B = np.random.uniform(-1, 1, size=[1, self.n_units])
        self.w_grads_mem = None
        self.b_grads_mem = None

    def set_weights(self, weights):
        w, b, w_grad_m, b_grad_m = weights
        w = np.copy(w)
        b = np.copy(b)
        assert w.shape == self.W.shape and b.shape == self.B.shape, 'dense layer weight not match'

        self.W = w
        self.B = b
        self.w_grads_mem = [np.copy(m) for m in w_grad_m] if w_grad_m else None
        self.b_grads_mem = [np.copy(b) for b in b_grad_m] if b_grad_m else None

    def get_weights(self):
        return [self.W, self.B, self.w_grads_mem, self.b_grads_mem]

    def Forward(self, inputs: np.ndarray):
        self.last_inputs = inputs
        y = inputs.dot(self.W) + self.B
        return y

    def Backward(self, output_gradients: np.ndarray, optimizer: Optimizer):
        w_grads = self.last_inputs.T.dot(output_gradients)
        b_grads = output_gradients.sum(axis=0, keepdims=True)
        x_grads = output_gradients.dot(self.W.T)

        w_result_grads, w_memory = optimizer.CalculateGradients(
            self.w_grads_mem, w_grads)
        b_result_grads, b_memory = optimizer.CalculateGradients(
            self.b_grads_mem, b_grads)

        self.w_grads_mem = w_memory
        self.b_grads_mem = b_memory

        self.W -= w_result_grads
        self.B -= b_result_grads
        return x_grads
