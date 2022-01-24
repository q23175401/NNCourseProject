from .base_optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=1e-4) -> None:
        self.learning_rate = learning_rate

    def CalculateGradients(self, last_grads_memory, new_grads):
        '''
        returns
            first: 
                the gradients can be applied
            second:
                the things the layer need to remember
        '''
        return self.learning_rate * new_grads, None
