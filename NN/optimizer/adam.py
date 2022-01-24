from .base_optimizer import Optimizer
import numpy as np

# Exponential moving average


class Adam(Optimizer):
    def __init__(self, learning_rate=1e-4, beta1=0.9, beta2=0.998) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def CalculateGradients(self, last_grads_memory, new_grads):
        if last_grads_memory is None:
            exp_beta1, exp_beta2 = 1, 1
            moments = np.zeros_like(new_grads)
            values = np.zeros_like(new_grads)
        else:
            moments, values, exp_beta1, exp_beta2 = last_grads_memory

        moments = self.beta1*moments + (1-self.beta1) * new_grads
        values = self.beta2*values + (1-self.beta2) * new_grads**2

        exp_beta1 *= self.beta1
        exp_beta2 *= self.beta2

        moments_hat = moments / (1-exp_beta1)
        values_hat = values / (1-exp_beta2)

        result_grads_output = self.learning_rate * \
            moments_hat / (np.sqrt(values_hat) + 1e-10)

        output_memory = [moments, values, exp_beta1, exp_beta2]
        return result_grads_output, output_memory
