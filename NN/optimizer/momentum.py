from .base_optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, learning_rate=1e-4, momentum_ratio=0.1) -> None:
        self.learning_rate = learning_rate
        self.momentum_ratio = momentum_ratio

    def CalculateGradients(self, last_grads_memory, new_grads):
        if last_grads_memory is not None:
            result_grads_output = self.momentum_ratio * last_grads_memory + \
                new_grads * self.learning_rate

        else:
            result_grads_output = self.learning_rate * new_grads

        output_memory = result_grads_output

        return result_grads_output, output_memory
