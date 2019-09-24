"""
Defines a model in Denserflow. A model is a full neural network,
and is comprised of a set of layers with associated loss function
and activation functions. Learning happens online.

Unlike other modules, there is only one model class, since the
user composes their own models.
"""
import logging
import random
from typing import List, Callable

import numpy as np
from nptyping import Array

from .activation import softmax
from .layer import Layer
from .error import Loss, CrossEntropyWithSoftmax

logger = logging.getLogger("DenserFlow.Model")


class Model:
    """
    Represents a neural network model.
    """

    def __init__(self, loss_func: Loss):
        """
        :param loss_func: The loss funcation to be used for this model.
        """
        # initialize layers
        self.layers = []
        self.params = []
        self.error_func = loss_func

    def add_layer(self, layer: Layer) -> None:
        """
        Add a layer to the model.
        :param layer: The layer to add.
        """
        if self.layers:
            layer._add_prev_layer(
                self.layers[-1].get_activation(), self.layers[-1].out_dim
            )
        self.layers.append(layer)

    def forward(self, input_batch: Array[float], mode: str = "train") -> Array[float]:
        """
        Perform a forward pass on the network for training.
        :param input_batch: The minibatched input. Must have shape (batch_size,)
        :param mode: Whether this is being used in train or test mode.
        """
        for layer in self.layers:
            output = layer.forward(input_batch, mode)
            input_batch = output
        return output

    def backward(self, delta: Array[float]) -> None:
        """
        Perform a backward pass on the network, given the delta of the error function.
        :param delta: The derivative of the error function of network with
        respect to the net of the output layer.
        """
        delta = self.layers[-1].backward(delta)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def update(self, lr: float, wd: float = 0, m: float = 0) -> None:
        """
        Updates the weights in the network, using gradients calculated
        in a backward pass.
        :param lr: learning rate to use
        :param wd: weight decay rate to use
        :param m: momentum rate to use
        """
        for layer in self.layers:
            layer.update(lr, wd, m)

    def update_adam(
        self, t: float, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999
    ) -> None:
        """
        Updates the weights in the network, using gradients calculated
        in a backward pass.
        :param lr: learning rate to use
        :param wd: weight decay rate to use
        :param m: momentum rate to use
        """
        for layer in self.layers:
            layer.update_adam(t, alpha, beta1, beta2)

    def make_batch(
        self,
        x: Array[float],
        y: Array[float],
        minibatch_size: int,
        shuffle: bool = True,
    ) -> List[Array[float]]:
        """
        Makes minibatches from given data, and optionally shuffles them.

        :param x: array of sample inputs
        :param y: array of sample labels
        :param minibatch_size: size of minibatches
        :param shuffle: Whether to shuffle the minibatches or not.
        """
        minibatches = []
        for idx in range(len(x) // minibatch_size):
            x_mini = []
            y_mini = []
            for i in range(idx, idx + minibatch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                x_mini.append(x[i])
                y_mini.append(y[i])
            minibatches.append((np.array(x_mini), np.array(y_mini)))
        if shuffle:
            random.shuffle(minibatches)
        return minibatches

    def SGD(
        self,
        x: Array[float],
        y: Array[float],
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        minibatch_size: int = 1,
        epochs: int = 100,
        adam: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        callback: Callable[["Model", int], float] = None,
    ) -> Array[float]:
        """
        Online learning with stochastic gradient descent. Returns an
        array of loss values at each epoch. If adam is true, then adam
        is used to optimise the network. A callback may also be passed in.
        At the end of each epoch, the callback will be called with the epoch
        number and the model object.

        :param x: Input data or features
        :param y: Input targets
        :param learning_rate: the learning rate used
        :param weight_decay: rate of weight decay used
        :param momentum: rate of momentum used
        :param epochs: number of times the dataset is presented to the
        network for learning
        :param: adam: If true, adam is used to optimise the network.
        :param beta1: Beta 1 value used in Adam optimisation.
        :param beta2: Beta 2 value used in Adam optimisation.
        :param callback: a callback function, provided the model itself.
        """
        x = np.array(x)
        y = np.array(y)
        loss_vals = np.zeros(epochs)

        # train!
        for k in range(epochs):
            theta = None
            loss = np.zeros(x.shape[0])

            minibatches = self.make_batch(x, y, minibatch_size)

            for x_mini, y_mini in minibatches:
                # ensure we have our minibatch sizes
                x_mini.reshape((minibatch_size, -1))
                y_mini.reshape((minibatch_size, -1))
                # make our predictions
                y_hat = self.forward(x_mini)
                # calculate loss and deltas
                loss, theta = self.error_func.calc_loss(
                    y_mini, y_hat, self.layers[-1].get_activation().f_deriv
                )
                # calculate our weights and then update
                self.backward(theta)

                if adam:
                    self.update_adam(k + 1, learning_rate, beta1, beta2)
                else:
                    self.update(learning_rate, weight_decay, momentum)

                loss_vals[k] = np.mean(loss)
            logger.info("epoch " + str(k + 1) + " loss: " + str(loss_vals[k]))
            # run callback if we have it
            if callback:
                callback(self, k)

        return loss_vals

    def predict(self, x: Array[float]) -> Array[float]:
        """
        Get the predictions of the model on a set of inputs.
        :param x: batch of inputs to feed into the model. Must have shape (batch_size,)
        """
        output = np.zeros((x.shape[0], self.layers[-1].out_dim))
        # for each sample
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :].reshape(1, -1), mode="test")
            # special case - we need to apply softmax without the loss function
            if type(self.error_func) is CrossEntropyWithSoftmax:
                output[i] = softmax().f(output[i])
        return output
