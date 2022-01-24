import numpy as np


class Perceptron():
    def __init__(self, input_units):
        self.w = np.random.uniform(-1, 1, [input_units, 1])
        self.b = np.random.uniform(-1, 1, [1, 1])

    def Predict(self, inputs):
        y = self.Forward(inputs)
        r = np.zeros_like(y)
        r[y >= 0] = 1
        return r

    def Forward(self, inputs):
        y = inputs @ self.w + self.b
        return y

    def Train(self, inputs, answers, learning_rate=0.1, epochs=20):

        for _ in range(epochs):
            for x, a in zip(inputs, answers):
                r = self.Predict(x)

                error = a - r[0, 0]
                self.w += learning_rate * error * x.reshape([len(x), 1])
                self.b += learning_rate * error
