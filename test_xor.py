import numpy as np
from NN.layer.activation.softmax import Softmax
from NN.loss.categorical_cross_entropy import CategoricalCrossEntropy
from NN.loss.least_squared_error import LeastSquaredError
from NN.network import Network
from NN.layer import DenseLayer
from NN.layer.activation import Sigmoid, Tanh


if __name__ == "__main__":
    # test for xor problem and multiple outputs
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    learning_rate = 1e-2

    myNet = Network([
        DenseLayer(2, 8),
        Tanh(),
        # Sigmoid(),
        DenseLayer(8, 5),
        Tanh(),
        # Sigmoid(),
        DenseLayer(5, 2),
        Softmax(),
        # Tanh(),
        # Sigmoid(),
    ], learning_rate,
        CategoricalCrossEntropy(),
        # LeastSquaredError(),
    )

    # training
    myNet.Train(x, y, batchsize=2, epochs=1000, verbose=0)

    # testing
    predictions = myNet.Predict(x)  # .argmax(axis=1)
    for p, a, i in zip(predictions, y, x):
        print('pre, ans, x', p, a, i)

