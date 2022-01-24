from .base_optimizer import Optimizer


# Exponential moving average
class EMA(Optimizer):
    def __init__(self, learning_rate=1e-4, new_grads_ratio=0.9) -> None:
        self.learning_rate = learning_rate
        self.new_grads_ratio = new_grads_ratio

    def CalculateGradients(self, last_grads_memory, new_grads):
        if last_grads_memory is not None:

            result_grads_output = (1-self.new_grads_ratio) * last_grads_memory + \
                self.learning_rate * self.new_grads_ratio * new_grads
        else:
            result_grads_output = self.learning_rate * new_grads

        output_memory = result_grads_output
        return result_grads_output, output_memory


# calcualte average gradients of all, and sum new gradient with new_grads_ratio
class AverageVector(Optimizer):
    def __init__(self, learning_rate=1e-4, new_grads_ratio=0.9) -> None:
        self.learning_rate = learning_rate
        self.new_grads_ratio = new_grads_ratio

    def CalculateGradients(self, last_grads_memory, new_grads):
        if last_grads_memory is not None:
            last_vector = last_grads_memory[0]
            num_vector = last_grads_memory[1]+1
            output_vector = last_vector + new_grads

            result_grads_output = (
                (1-self.new_grads_ratio) * output_vector/num_vector + self.new_grads_ratio*new_grads) * self.learning_rate

            output_memory = [output_vector, num_vector]
        else:
            result_grads_output = self.learning_rate * new_grads

            output_memory = [new_grads, 1]

        return result_grads_output, output_memory
