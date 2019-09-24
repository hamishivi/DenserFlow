"""
Defines loss functions. Each loss function has one defined function,
calc_loss, which returns a two values: the loss and the delta. The
loss is simply the evaluation of the loss function given a set of
predictions and associated labels. Delta is a bit more complicated:
it is the value of the derivative of the error function with respect
to the predicted output of the neural network, times the derivative
of the output of the neural network with respect to the net of the
output layer. That is, the delta is the derivative of the loss with
respect to the net of the output layer.
"""
from typing import Callable, Tuple
import numpy as np
from nptyping import Array

from .activation import Activation, softmax


class Loss:
    """
    The parent class for all loss functions.
    """

    def calc_loss(
        self,
        y: Array[float],
        y_hat: Array[float],
        activation_deriv: Callable[[Array[float]], Array[float]],
    ) -> Tuple[Array[float], Array[float]]:
        """
        :param y: a symbolic tensor representing the
        set of true labels for the input to the neural network.
        :param y_hat: a symbolic tensor representing the
        predicted labels of the input to the neural network.
        :param activation_deriv: the derivative of the activation
        function of the last layer.
        """
        raise AttributeError("Unimplemented loss function")


class MSE(Loss):
    """
    The mean squared error loss
    """

    def calc_loss(
        self, y: Array[float], y_hat: Array[float], activation_deriv: Activation
    ) -> Tuple[Array[float], Array[float]]:
        # activation_deriv is the last layer's deriv
        error = y_hat - y
        loss = error ** 2
        # take mean loss across each batch
        loss = 0.5 * np.mean(loss, axis=0)
        # calculate the delta of the output layer
        delta = -error * activation_deriv(y_hat)
        # return loss and delta
        return loss, delta


class CrossEntropy(Loss):
    """
    Cross entropy loss, without softmax. For best results,
    a softmax or logistic activation should be applied on
    the output layer. The true labels (y) should be one-hot
    encoded vectors.
    """

    def calc_loss(
        self, y: Array[float], y_hat: Array[float], activation_deriv: Activation
    ) -> Tuple[Array[float], Array[float]]:
        batch_size = y_hat.shape[0]
        y_hat += 1e-18  # add a small epsilon to avoid divide by 0s
        # calculate cross entropy loss, assuming y is one-hot encoded
        loss = -np.sum(np.multiply(y, np.log(y_hat))) / batch_size
        # calculate the deltas of the output layer, using matrix multiplication
        deltas = np.zeros((batch_size, y_hat.shape[1]))
        idx = 0
        for answer, predict in zip(y, y_hat):
            deltas[idx] = (-answer / predict) @ activation_deriv(predict)
            idx += 1
        # return loss and delta
        return loss, deltas


class CrossEntropyWithSoftmax(Loss):
    """
    Cross entropy loss with softmax included. For best results,
    do not place an activation function on the output layer when
    using this loss function. The true labels (y) should be one-hot
    encoded vectors.
    """

    def __init__(self):
        self.softmax = softmax()

    def calc_loss(
        self, y: Array[float], y_hat: Array[float], activation_deriv: Activation
    ) -> Tuple[Array[float], Array[float]]:
        batch_size = y_hat.shape[0]
        # run softmax on each set of predictions
        activations = np.apply_along_axis(self.softmax.f, 1, y_hat)
        # calculate cross entropy loss, assuming y is one-hot encoded
        loss = -np.sum(np.multiply(y, np.log(activations + 1e-18))) / batch_size
        # calculate the delta of the output layer, using matrix multiplication
        delta = activations - y
        # return loss and delta
        return loss, delta
