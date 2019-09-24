"""
This file defines activation functions for layers of
the neural network. Each function has its base and first
derivative defined. Some activation functions
have associated hyperparameters, and must be initialised with them.
"""
import numpy as np
from nptyping import Array


class Activation(object):
    """
    All activation functions inherit from this parent class.
    """

    def f(self, x: Array[float]) -> Array[float]:
        """
        :param x: a symbolic tensor representing one input to the activation function.
        """
        raise AttributeError("Unimplemented activation function")

    def f_deriv(self, a: Array[float]) -> Array[float]:
        """
        :param a: a symbolic tensor representing one input
        to the derivation of the activation function.
        """
        raise AttributeError("Unimplemented derivation of activation function")


class linear(Activation):
    """
    A linear activation function, f(x) = x.
    """

    def f(self, x: Array[float]) -> Array[float]:
        return x

    def f_deriv(self, a: Array[float]) -> Array[float]:
        return np.zeros_like(a) + 1


class tanh(Activation):
    """
    The tanh activation function, f(x) = tanh(x).
    """

    def f(self, x: Array[float]) -> Array[float]:
        return np.tanh(x)

    def f_deriv(self, a: Array[float]) -> Array[float]:
        return 1.0 - a ** 2


# This util func prevents overflows by catching
# large values and returning values for them.
# These are 'close enough' in most cases.
# we then vectorise it for speed.
def _safe_logistic(x: float) -> float:
    if x > 500:
        return 1.0
    elif x < -500:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


_vec_log = np.vectorize(_safe_logistic)


class logistic(Activation):
    def f(self, x: Array[float]) -> Array[float]:
        """
        The logistic or sigmoid function, f(x) = logistic(x).
        """
        return _vec_log(x)

    def f_deriv(self, a: Array[float]) -> Array[float]:
        return a * (1 - a)


class relu(Activation):
    """
    The rectified linear unit function. f(x) = max(0, x).
    """

    def f(self, x: Array[float]) -> Array[float]:
        return np.where(x > 0, x, 0)

    def f_deriv(self, a: Array[float]) -> Array[float]:
        return np.where(a > 0, 1, 0)


class leaky_relu(Activation):
    """
    The 'leaky' rectified linear unit function. f(x) = max(0, x).
    Takes in the alpha (gradient of line when x < 0) at initialisation.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def f(self, x: Array[float]) -> Array[float]:
        return np.where(x > 0, x, x * self.alpha)

    def f_deriv(self, a: Array[float]) -> Array[float]:
        return np.where(a > 0, 1, self.alpha)


# utility function, since the limit of the derivative
# when x -> + infinity is 1, and when x -> -infinity
# is 0.
def _safe_gelu_deriv(x: float) -> float:
    if x > 200:
        return 1.0
    elif x < -200:
        return 0.0
    exp = np.exp(1.702 * x)
    exp_plus = exp + 1
    return exp * (1.702 * x + exp_plus) / ((exp_plus) ** 2)


_gelu_deriv_vec = np.vectorize(_safe_gelu_deriv)


class gelu(Activation):
    """
    A simple approximation of the gaussian linear unit, or 'gelu'.
    Approximation from https://arxiv.org/pdf/1606.08415.pdf.
    """

    def f(self, x: Array[float]) -> Array[float]:
        return np.multiply(x, _vec_log(1.702 * x))

    def f_deriv(self, a: Array[float]) -> Array[float]:
        return _gelu_deriv_vec(a)


class softmax(Activation):
    """
    The softmax activation function. Vectorised derivative from
    https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function.
    """

    def f(self, x: Array[float]) -> Array[float]:
        # np.max to normalise a bit
        # idea from http://cs231n.github.io/linear-classify/#softmax
        # subtracting the max improves stability.
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def f_deriv(self, a: Array[float]) -> Array[float]:
        s = a.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
