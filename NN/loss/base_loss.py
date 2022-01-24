import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def Loss(self, network_y, answers):
        raise NotImplementedError()

    @abstractmethod
    def Gradients(self, network_y, answers):
        raise NotImplementedError()
