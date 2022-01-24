import numpy as np
from .base_activation import Activation


class Softmax(Activation):
    def func(self, inputs):
        max_value = inputs.max(axis=1, keepdims=True)
        all_exp = np.exp(inputs-max_value)
        sum_exp = np.sum(all_exp, axis=1, keepdims=True)
        return all_exp / sum_exp

    def derivative_func(self, inputs):
        # calculate jacobian matrix
        # [Si(1*int(i==j) - SiSj)] shape = (j x j)
        all_jacobian = []
        y_preds = self.func(inputs)

        # reshape shape = [batch, 1, j]
        y_preds = y_preds.reshape([y_preds.shape[0], 1, y_preds.shape[-1]])
        for y_pred in y_preds:
            all_identity = y_pred * np.identity(y_pred.size)
            all_pairs = y_pred.T @ y_pred
            jacobian_matrix = all_identity - all_pairs
            all_jacobian.append(jacobian_matrix)

        return all_jacobian

    def Backward(self, output_gradients, _optimizer):
        all_jacobian = self.derivative_func(self.last_inputs)

        all_dE_dx = []
        for dE_dy, j_matrix in zip(output_gradients, all_jacobian):
            dE_dx = dE_dy @ j_matrix
            all_dE_dx.append(dE_dx)

        x_gradients = np.array(all_dE_dx)
        return x_gradients
