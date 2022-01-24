import math as m
import numpy as np
from .loss import Loss, MeanSquaredError
from .optimizer import Optimizer, SGD, Adam


class RBFNetwork:
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        kmeans_iter=20,
        loss_func: Loss = MeanSquaredError(),
        optimizer: Optimizer = SGD(learning_rate=1e-3),
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_func = loss_func
        self.optimizer = optimizer

        # create parameters
        self.input = np.zeros(self.input_dim)
        self.centers = np.zeros((self.hidden_dim, self.input_dim))
        self.stds = np.zeros(self.hidden_dim)
        self.hidden_output = np.zeros(self.hidden_dim)

        self.w_grads_mem = None
        self.b_grads_mem = None
        self.s_grads_mem = None
        self.c_grads_mem = None

        self.kmeans_iteration = kmeans_iter
        self.set_centers_before = False
        self.W = np.random.uniform(0, 1, [self.hidden_dim, self.output_dim])
        self.B = np.random.uniform(0, 1, [1, self.output_dim])

    def kmeans(self, iteration, centers, inputs):
        k = len(centers)
        for _ in range(iteration):
            clusters = [[] for _ in range(k)]
            for input in inputs:
                clusters[((centers - input) ** 2).sum(axis=1).argmin()].append(input)

            for c in range(k):
                if len(clusters[c]) != 0:
                    centers[c] = np.mean(clusters[c], axis=0)

        return centers

    def setup_centers(self, inputs=None):
        """Setup center using clustering"""
        self.centers = np.random.uniform(
            size=[self.hidden_dim, self.input_dim],
            low=0,
            high=1,
        )
        # using kmeans to initializer centers
        if inputs is not None:
            self.centers = self.kmeans(self.kmeans_iteration, self.centers, inputs)

        self.setup_std_for_centers()

    def setup_std_for_centers(self):
        for i in range(self.hidden_dim):
            center = self.centers[i]
            self.stds[i] = self.calculate_center_stds(center)

    def calculate_center_stds(self, center):
        distances = np.zeros([self.hidden_dim])
        for i in range(self.hidden_dim):
            distances[i] = self.euclidean_distance(center, self.centers[i])
        min_dist_idx = np.argsort(distances)

        sum = 0
        # get nearest few neighbors
        p = self.hidden_dim // 3 if self.hidden_dim >= 6 else self.hidden_dim
        for i in range(p):
            nearest = min_dist_idx[i]

            neighbour_centroid = self.centers[nearest]
            sum += np.sum((center - neighbour_centroid) ** 2)

        sigma = sum / p
        sigma = m.sqrt(sigma)
        return sigma

    @staticmethod
    def euclidean_distance(x, y):
        return np.linalg.norm(x - y)

    # train n epochs
    def train(self, inputs, targets, epochs, verbose=1):
        if not self.set_centers_before:
            self.setup_centers(inputs)
            self.set_centers_before = True

        for i in range(epochs):

            error = self.train_one_epoch(inputs, targets)

            if verbose > 0:
                print("Iteration ", i, " Error ", error)

        return error

    # Train an epoch and return total MSE
    def train_one_epoch(self, inputs, targets):
        # print("Pass one epoch")
        all_error = 0
        all_index = np.arange(len(inputs))
        np.random.shuffle(all_index)

        for random_index in all_index:
            input = inputs[random_index]
            target = targets[random_index]

            output = self.forward(input)
            self.backward(output, target)

            all_error += self.loss_func.Loss(np.array([output]), np.array([target])).mean()

        all_error = all_error / len(inputs)
        return all_error

    def forward(self, input):
        """hidden radial function layer"""
        hidden_output = np.zeros([self.hidden_dim])
        for i in range(self.hidden_dim):
            # pass input to gaussian function
            dist_sqared = self.euclidean_distance(input, self.centers[i]) ** 2
            hidden_output[i] = 1 / (self.stds[i] * m.sqrt(2 * m.pi)) * m.exp(-(dist_sqared / (2 * self.stds[i] ** 2)))

        """ output weighting layer """
        output = (np.array([hidden_output]) @ self.W + self.B)[0]

        total = output.sum()
        output /= total
        # carry parameters
        self.input = input
        self.hidden_output = hidden_output
        return output

    # Weight update by gradient descent algorithm
    def backward(self, output, target):
        # self.error_of_output_layer = np.zeros([self.output_dim])
        output_gradients = self.loss_func.Gradients(np.array([output]), np.array([target]))

        """pass gradients to weighting layer"""
        # Adjust hidden to output weight and bias
        w_grads = np.array([self.hidden_output]).T @ output_gradients
        b_grads = output_gradients
        x_grads = output_gradients @ self.W.T

        """pass gradients to radial function layer"""
        # Adjust center and std, input to hidden weight
        c_grads = np.zeros_like(self.centers)
        s_grads = np.zeros_like(self.stds)

        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                x_grad = x_grads[0, j]

                c_second_part = (
                    (self.input[i] - self.centers[j][i]) / (m.pow(self.stds[j], 3) + 1e-10) / m.sqrt(2 * m.pi)
                )
                c_grad = self.hidden_output[j] * c_second_part * x_grad

                std_second_part = (m.pow((self.input[i] - self.centers[j][i]), 2)) / (
                    m.pow(self.stds[j], 4) + 1e-10
                ) / m.sqrt(2 * m.pi) - 1 / (self.stds[j] + 1e-10) / m.sqrt(2 * m.pi)
                s_grad = self.hidden_output[j] * std_second_part * x_grad

                c_grads[j, i] += c_grad
                s_grads[j] += s_grad

        w_result_grads, self.w_grads_mem = self.optimizer.CalculateGradients(self.w_grads_mem, w_grads)
        b_result_grads, self.b_grads_mem = self.optimizer.CalculateGradients(self.b_grads_mem, b_grads)
        c_result_grads, self.c_grads_mem = self.optimizer.CalculateGradients(self.c_grads_mem, c_grads)
        s_result_grads, self.s_grads_mem = self.optimizer.CalculateGradients(self.s_grads_mem, s_grads)

        self.W -= w_result_grads
        self.B -= b_result_grads
        self.stds -= s_result_grads
        self.centers -= c_result_grads

    def get_accuracy(self, inputs, targets, predict_func=None):
        correct = 0
        for i in range(len(inputs)):
            pattern = inputs[i]
            target = targets[i]
            output = self.forward(pattern)

            if predict_func:
                predict_label = predict_func(output)
                target_label = predict_func(target)
            else:
                predict_label = np.argmax(output)
                target_label = np.argmax(target)

            if predict_label == target_label:
                correct += 1
        accuracy = (float)(correct / len(inputs))
        return accuracy


def use_example():
    """Create xor data"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    rbf = RBFNetwork(2, 10, 2, loss_func=MeanSquaredError(), optimizer=Adam(1e-3))
    mse = rbf.train(X, y, 1500, verbose=0)
    print("Last MSE ", mse)

    accuracy = rbf.get_accuracy(X, y)
    print("Total accuracy is ", accuracy)


if __name__ == "__main__":
    use_example()
