from NN.optimizer.base_optimizer import Optimizer
from NN.optimizer.sgd import SGD
from .loss.least_squared_error import LeastSquaredError
from .loss.base_loss import Loss
import numpy as np
import time
import pickle


class Network():
    def __init__(self, layer_list, optimizer: Optimizer = SGD(1e-4), loss_func: Loss = LeastSquaredError()) -> None:
        self.layerStack = layer_list
        self.optimizer = optimizer
        self.LossFunc = loss_func

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def Predict(self, inputs, select_prediction_func=None):
        y = self.Forward(inputs)

        if select_prediction_func is not None:
            y = select_prediction_func(y)

        return y

    def Forward(self, inputs):
        x = np.array(inputs)
        for l in self.layerStack:
            x = l.Forward(x)

        return x

    def Backward(self, output_gradents):
        x_gradients = output_gradents
        for l in reversed(self.layerStack):
            x_gradients = l.Backward(x_gradients, self.optimizer)

    def getNextRandomBatch(self, inputs, answers, batchsize):
        batches = list(range(0, len(answers), batchsize))
        np.random.shuffle(batches)
        for b in batches:
            yield inputs[b:b+batchsize], answers[b:b+batchsize]

    def train_on_batch(self, batch_inputs, ans):
        y = self.Forward(batch_inputs)
        output_gradients = self.LossFunc.Gradients(y, ans)
        self.Backward(output_gradients)

        err = self.LossFunc.Loss(y, ans)
        return err.mean()

    def Train(self, inputs, answers, batchsize=8, epochs=20, train_acc=False, verbose=2, select_prediction_func=None):

        for ep in range(epochs):
            total_err = 0

            start_time = time.time()
            for bx, by in self.getNextRandomBatch(inputs, answers, batchsize):
                total_err += self.train_on_batch(bx, by)
            time_passed = time.time() - start_time

            if verbose > 0:
                total_batches = (len(inputs)+batchsize-1) // batchsize

                verbose_string = f'epochs {ep} loss {total_err/total_batches if total_batches>0 else 0}'

                if verbose > 1 and select_prediction_func is not None and train_acc:
                    pre_y = self.Predict(inputs, select_prediction_func)
                    pre_a = select_prediction_func(answers)

                    accuracy = self.getAccuracy(pre_y, pre_a)

                    verbose_string += " training accuracy: " + str(accuracy)

                if verbose > 2:
                    verbose_string += " time comsumed an epoch " + \
                        str(time_passed)
                print(verbose_string)

    def getAccuracy(self, predictions, answers):
        correct = 0
        for py, pa in zip(predictions, answers):
            correct += 1 if py == pa else 0

        accuracy = correct / len(predictions)if len(predictions) != 0 else 0
        return accuracy

    def get_weights(self):
        weights_list = []

        for l in self.layerStack:
            weights_list.append(l.get_weights())

        return weights_list

    def set_weights(self, weights_list):
        for w, l in zip(weights_list, self.layerStack):
            l.set_weights(w)

    def save_network(self, filename='mynet.pickle'):

        with open(file=filename, mode='wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_network(filename='mynet.pickle'):

        with open(file=filename, mode='rb') as f:
            mynet = pickle.load(f)
        return mynet
