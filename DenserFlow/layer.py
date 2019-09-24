"""
Defines neural network layers. A layer in a network
has various attributes - see the Layer parent class
for a description of each main function.
"""
from typing import Optional
import numpy as np
from nptyping import Array

from .activation import Activation, linear, logistic, relu, leaky_relu


class Layer(object):
    """
    The parent class for all layer classes.
    """

    def __init__(self, activation: Activation = linear()):
        self.prev_activation = None
        self.activation = activation

    def _add_prev_layer(self, prev_activation: Activation, prev_out_dim: int) -> None:
        """
        Utility function used when creating a model. For our
        implementation of backpropagation, the activation function
        of the previous layer in the network must be known. This
        should not be called by the end user.
        """
        self.prev_activation = prev_activation
        self.in_dim = prev_out_dim

    def get_activation(self) -> Activation:
        """
        Returns activation function for this layer.
        """
        return self.activation

    def forward(self, input_batch: Array[float], mode: str = "train") -> Array[float]:
        """
        Performs a forward pass given an input batch.
        :param input_batch: the input to the layer
        :param mode: whether we are training or testing the layer at this point.
        """
        raise AttributeError("Unimplemented layer function")

    def backward(self, deltas: Array[float]) -> Array[float]:
        """
        Performs a backward pass given a delta.
        :param deltas: the delta passed through from the previous layer -
        the derivative of the error function with respect to the
        net of the current layer.
        """
        raise AttributeError("Unimplemented layer function")

    def update(self, lr: float, wd: float, m: float) -> None:
        """
        Update the weights of this layer.
        :param lr: the learning rate of the model.
        :param wd: the weight decay rate of the model.
        :param m: the momentum rate of the model.
        """
        raise AttributeError("Unimplemented layer functions")

    def update_adam(self, t: int, alpha: float, beta1: float, beta2: float) -> None:
        """
        Update the weights of this layer using the adam algorithm.
        :param t: the current timestep of the model (usually the current epoch number)
        :param alpha: the learning rate of the model.
        :param beta1: the decay rate for the first moment.
        :param beta2: the decay rate for the second moment.
        """
        raise AttributeError("Unimplemented layer functions")


class DenseLayer(Layer):
    """
     A fully connected layer. Weight matrix W should be of shape (n_in,n_out)
     and the bias vector b should be of shape (n_out,).

    Hidden unit activation is given by: f(dot(input,W) + b)

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Activation = linear(),
        W: Optional[Array[float]] = None,
        b: Optional[Array[float]] = None,
    ):
        """
        :param n_in: dimensionality of input

        :param n_out: number of hidden units

        :param activation: Non linearity to be applied in the hidden layer

        :param W: Weight matrix to initialise weights with.
        If none, Xavier Initialisation is used.

        :param b: Bias matrix to initialise bias values with.
        If none, Xavier Initialisation is used.
        """
        super().__init__(activation)
        # useful variable to store
        self.in_dim = n_in
        self.out_dim = n_out
        self.input = None

        # initialise weights and bias
        if W is None:
            # we use He initialisation if using relu
            if type(activation) is relu or type(activation) is leaky_relu:
                self.W = np.random.normal(
                    loc=0, scale=np.sqrt(2.0 / n_in), size=(n_in, n_out)
                )
            # else use xavier initialisation
            else:
                self.W = np.random.uniform(
                    low=-np.sqrt(6.0 / (n_in + n_out)),
                    high=np.sqrt(6.0 / (n_in + n_out)),
                    size=(n_in, n_out),
                )
            if type(activation) is logistic:
                self.W *= 4

        self.b = np.zeros((n_out))

        # initialise gradient matrices
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        self.prev_grad_W = np.zeros(self.W.shape)
        self.prev_grad_b = np.zeros(self.b.shape)

        self.m_W = np.zeros(self.W.shape)
        self.v_W = np.zeros(self.W.shape)

        self.m_b = np.zeros(self.b.shape)
        self.v_b = np.zeros(self.b.shape)

    # utility function for forward pass below
    def _apply_weights(self, input_row):
        return np.dot(input_row, self.W)

    def forward(self, input_batch: Array[float], mode: str = "train") -> Array[float]:
        output = np.zeros((input_batch.shape[0], self.out_dim))
        # iterate over each sample in the minibatch
        lin_output = np.apply_along_axis(self._apply_weights, 1, input_batch) + self.b
        output = np.apply_along_axis(self.activation.f, 1, lin_output)
        # save values for backward pass
        self.output = output
        self.input = input_batch
        return self.output

    def backward(self, deltas: Array[float]) -> Array[float]:
        # save previous gradients
        self.prev_grad_W = self.grad_W
        self.prev_grad_b = self.grad_b
        # accumulate gradients over the minibatch
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        # prepare for next layer
        next_deltas = np.zeros((deltas.shape[0], self.in_dim))
        idx = 0
        self.grad_b += np.sum(deltas)
        for delta, sample in zip(deltas, self.input):
            self.grad_W += np.atleast_2d(sample).T.dot(np.atleast_2d(delta))
            if self.prev_activation is not None:
                next_deltas[idx] = delta.dot(self.W.T) * self.prev_activation.f_deriv(
                    sample
                )
            idx += 1
        return next_deltas

    def update_adam(self, t: int, alpha: float, beta1: float, beta2: float) -> None:
        # adam code based off the fourth tutorial of comp5329. Adapted for our purposes.
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.grad_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.grad_W ** 2)
        m_hat = self.m_W / (1 - beta1 ** t)
        v_hat = self.v_W / (1 - beta2 ** t)
        self.W = self.W - alpha * (m_hat / (np.sqrt(v_hat) - 1e-8))

        self.m_b = beta1 * self.m_b + (1 - beta1) * self.grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.grad_b ** 2)
        m_hat = self.m_b / (1 - beta1 ** t)
        v_hat = self.v_b / (1 - beta2 ** t)
        self.b = self.b - alpha * (m_hat / (np.sqrt(v_hat) - 1e-8))

    def update(self, lr: float, wd: float, m: float) -> None:
        self.W -= (lr * (self.grad_W + m * self.prev_grad_W)) - (lr * wd * self.grad_W)
        self.b -= (lr * (self.grad_b + m * self.prev_grad_b)) - (lr * wd * self.grad_b)


class ActivationLayer(Layer):
    """
    A layer that just applies an activation function. This can be used if you
    wish to apply an activation independent of a dense layer.
    """

    def __init__(self, activation: Activation):
        """
        :param activation: The activation function to use.
        """
        self.activation = activation

    # we dont change the dimensions of the data in an activation layer,
    # so in dim == out dim.
    def _add_prev_layer(self, prev_activation: Activation, prev_out_dim: int) -> None:
        self.prev_activation = prev_activation
        self.in_dim = prev_out_dim
        self.out_dim = prev_out_dim

    def forward(self, input_batch: Array[float], mode: str = "train") -> Array[float]:
        output = np.apply_along_axis(self.activation.f, 1, input_batch)
        # save values for backward pass
        self.output = output
        self.input = input_batch
        return output

    def backward(self, deltas: Array[float]) -> Array[float]:
        # prepare for next layer
        next_deltas = np.zeros((deltas.shape[0], self.in_dim))
        idx = 0
        for delta, sample in zip(deltas, self.input):
            if self.prev_activation is not None:
                next_deltas[idx] = delta * self.prev_activation.f_deriv(sample)
            idx += 1
        return next_deltas

    def update(self, lr: float, wd: float, m: float) -> None:
        pass

    def update_adam(self, t: int, alpha: float, beta1: float, beta2: float) -> None:
        pass


class Dropout(Layer):
    """
    A Dropout layer. This randomly drops nodes in the previous
    layer at a given probability rate.
    """

    def __init__(self, dropout_rate: float):
        """
        :param dropout_rate: The rate at which units should be dropped.
        Must be between 0 and 1.
        """
        super().__init__()
        self.keep_rate = 1 - dropout_rate

    # we dont change the dimensions of the data in a dropout layer,
    # so in dim == out dim.
    def _add_prev_layer(self, prev_activation: Activation, prev_out_dim: int) -> None:
        self.prev_activation = prev_activation
        self.in_dim = prev_out_dim
        self.out_dim = prev_out_dim

    def forward(self, input_batch: Array[float], mode: str = "train") -> Array[float]:
        self.dropout_matrix = (
            np.random.binomial(1, self.keep_rate, size=input_batch.shape)
            / self.keep_rate
        )
        return np.reshape(input_batch * self.dropout_matrix, input_batch.shape)

    def backward(self, deltas: Array[float]) -> Array[float]:
        return deltas * self.dropout_matrix

    # nothing to learn, so nothing happens in update
    def update(self, lr: float, wd: float, m: float) -> None:
        pass

    def update_adam(self, t: int, alpha: float, beta1: float, beta2: float) -> None:
        pass


class BatchNormLayer(Layer):
    """
    A Batch Normalisation layer. Performs batch normalisation on
    the input for the layer directly after it.
    Utilises a moving mean and variance calculation for use in test:
    while training, the predicted mean and variance of the entire population
    is adjusted at each batch, and then these predictions are used at test time.
    Layer based on this article:
    https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
    Moving mean and variance from see:
    https://stats.stackexchange.com/questions/219808/how-and-why-does-batch-normalization-use-moving-averages-to-track-the-accuracy-o
    """

    def __init__(self, momentum: float = 0.9):
        """
        :param momentum: rate of momentum applied when computing
        moving mean and variance
        """
        super().__init__()
        # save appropriate values
        self.input = None
        self.momentum = momentum

        # initialise batchnorm params
        self.gamma = 1
        self.beta = 0
        # initialise values used for training
        self.grad_gamma = 0
        self.grad_beta = 0
        self.prev_grad_gamma = 0
        self.prev_grad_beta = 0
        # initialise moving mean and variance
        self.moving_mu = 0
        self.moving_var = 0
        # adam optimiser
        self.m_gamma = 0
        self.v_gamma = 0

        self.m_beta = 0
        self.v_beta = 0

    # we dont change the dimensions of the data in a batchnorm layer,
    # so in dim == out dim.
    def _add_prev_layer(self, prev_activation: Activation, prev_out_dim: int) -> None:
        self.prev_activation = prev_activation
        self.in_dim = prev_out_dim
        self.out_dim = prev_out_dim

    def forward(self, input_batch: Array[float], mode: str = "train") -> Array[float]:
        if mode == "train":
            # calc mu and var
            self.mu = np.mean(input_batch, axis=0)
            self.var = np.var(input_batch, axis=0)

            # batchnorm: compute moving mean and std for use in eval
            self.moving_mu = (
                self.momentum * self.moving_mu + (1 - self.momentum) * self.mu
            )
            self.moving_var = (
                self.momentum * self.moving_var + (1 - self.momentum) * self.var
            )
        else:
            # in test mode, we use the moving averages
            self.mu = self.moving_mu
            self.var = self.moving_var
        # normalise input
        self.X_norm = (input_batch - self.mu) / np.sqrt(self.var + 1e-8)
        # scale normalised input
        lin_output = self.X_norm * self.gamma + self.beta
        # we perform no activation on this layer: it is just batch norm-ing
        self.output = lin_output
        # cache input for backprop
        self.input = input_batch
        return self.output

    def backward(self, deltas: Array[float]) -> Array[float]:
        # code based off https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
        # but we have adapted it for our own purposes

        # cache previous gradients
        self.prev_grad_gamma = self.grad_gamma
        self.prev_grad_beta = self.grad_beta
        # calculate batchnorm gradients
        X_mu = self.input - self.mu
        std_inv = 1.0 / np.sqrt(self.var + 1e-8)

        dX_norm = deltas * self.gamma

        np.sum(dX_norm * X_mu, axis=0)
        dvar = np.sum(dX_norm * X_mu, axis=0) * -0.5 * (self.var + 1e-8) ** (-3 / 2)
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / self.in_dim) + (dmu / self.in_dim)
        dgamma = np.sum(deltas * self.X_norm, axis=0)
        dbeta = np.sum(deltas, axis=0)
        # save batch norm gradients
        self.grad_gamma = dgamma
        self.grad_beta = dbeta

        if self.prev_activation is not None:
            deltas = dX * self.prev_activation.f_deriv(self.input)
        return deltas

    def update(self, lr: float, wd: float, m: float) -> None:
        self.gamma -= (lr * (self.grad_gamma + m * self.prev_grad_gamma)) - (
            lr * wd * self.grad_gamma
        )
        self.beta -= (lr * (self.grad_beta + m * self.prev_grad_beta)) - (
            lr * wd * self.grad_beta
        )

    def update_adam(self, t: int, alpha: float, beta1: float, beta2: float) -> None:
        # adam code based off the fourth tutorial of comp5329. Adapted for our purposes.
        self.m_gamma = beta1 * self.m_gamma + (1 - beta1) * self.grad_gamma
        self.v_gamma = beta2 * self.v_gamma + (1 - beta2) * (self.grad_gamma ** 2)
        m_hat = self.m_gamma / (1 - beta1 ** t)
        v_hat = self.v_gamma / (1 - beta2 ** t)
        self.gamma = self.gamma - alpha * (m_hat / (np.sqrt(v_hat) - 1e-8))

        self.m_beta = beta1 * self.m_beta + (1 - beta1) * self.grad_beta
        self.v_beta = beta2 * self.v_beta + (1 - beta2) * (self.grad_beta ** 2)
        m_hat = self.m_beta / (1 - beta1 ** t)
        v_hat = self.v_beta / (1 - beta2 ** t)
        self.beta = self.beta - alpha * (m_hat / (np.sqrt(v_hat) - 1e-8))
