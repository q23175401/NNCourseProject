# my modules
from NN.optimizer import SGD
from NN.loss import CategoricalCrossEntropy, MeanSquaredError
from NN.layer.activation import Sigmoid, Softmax, Tanh, ReLU
from NN.layer import DenseLayer, Flatten, Conv2D, Reshape
from NN.network import Network

# libiaries
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from NN.optimizer.adam import Adam

from NN.optimizer.momentum import Momentum
from NN.optimizer.moving_average import EMA, AverageVector


def getOneHot(y_label):
    y = []
    for yl, i in enumerate(y_label):
        onehot_vector = np.zeros([10])
        onehot_vector[i] = 1
        y.append(onehot_vector)
    y = np.array(y)
    return y


if __name__ == "__main__":
    import time

    # load datas
    mnist = load_digits()
    x, tx, y_label, ty_label = train_test_split(
        mnist.images, mnist.target, test_size=0.2)
    x, tx = x/255.0, tx/255.0
    y = getOneHot(y_label)
    ty = getOneHot(ty_label)

    # build models

    # hyperparameters
    n_filters = 6
    kernel_len = 3
    stride_len = 2
    kernel_size = [kernel_len, kernel_len]
    stride_size = [stride_len, stride_len]
    dilation_size = [1, 1]
    padding = 'same'
    image_shape = [8, 8]

    net = Network([
        Flatten(image_shape),
        DenseLayer(64, 32),
        Tanh(),
        # Reshape([64], [8, 8, 1]),
        # Conv2D(input_shape=[8, 8, 1],
        #        n_filters=n_filters,
        #        kernel_size=kernel_size,
        #        strides=stride_size,
        #        dilation_rate=dilation_size,
        #        padding=padding),
        # Tanh(),
        # Conv2D(input_shape=[4, 4, n_filters],
        #        n_filters=n_filters*2,
        #        kernel_size=kernel_size,
        #        strides=stride_size,
        #        dilation_rate=dilation_size,
        #        padding=padding),
        # Tanh(),

        # Flatten([2, 2, 2*n_filters]),
        # DenseLayer(4*2*n_filters, 10),
        DenseLayer(32, 10),
        ReLU(),

        DenseLayer(10, 10),
        Softmax(),
    ],
        # optimizer=AverageVector(1e-3, 0.90),
        # optimizer=Momentum(1e-3, 0.8),
        # optimizer=EMA(1e-3, 0.95),
        # optimizer=SGD(1e-3),
        optimizer=Adam(1e-3, 0.9, 0.998),
        loss_func=MeanSquaredError(),
    )

    # training
    start_time = time.time()

    net.Train(x, y, batchsize=32, epochs=200, verbose=3)

    end_time = time.time()
    print('total time comsumed', end_time - start_time)

    # testing
    target = net.Predict(tx)
    ptx = target.argmax(axis=1)
    print('accuracy:', accuracy_score(ptx, ty_label))
    confusion_matrix(ptx, ty_label)

    # test save my network
    save_name = 'mynet_save/test_cnn_save1.pickle'
    net.save_network(save_name)

    net = Network.load_network(save_name)
    target2 = net.Predict(tx)

    print(np.allclose(target, target2))
