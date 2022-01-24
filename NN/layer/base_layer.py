import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def Forward(self, inputs: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def Backward(self, output_gradients, optimizer):
        raise NotImplementedError()

    def get_weights(self):
        return []

    def set_weights(self, weights):
        pass

    def get_config(self):
        return {}

    def __call__(self, inputs):
        return self.Forward(inputs)
