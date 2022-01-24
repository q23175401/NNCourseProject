from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def CalculateGradients(self, last_grads_memory, new_grads):
        '''
        returns
            result_grads_output:
                the gradients can be applied
            new_grads_memory:
                the things the layer need to remember, when optimize next time step
        '''
        # things a layer need to remember this moment
        result_grads_output = None
        new_grads_memory = [None]
        return result_grads_output, new_grads_memory
