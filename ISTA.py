from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
        X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.

    """

    bias_prime = bias - 2 * eta * np.sum(X @ weight + bias - y)

    weight -= 2 * eta * np.sum(X * (X @ weight + bias - y)[:, None], 0)

    weight_prime = np.zeros(len(weight))

    lesser = weight < -2 * eta * _lambda
    greater = weight > 2 * eta * _lambda

    weight_prime[lesser] = weight[lesser] + 2 * eta * _lambda
    weight_prime[greater] = weight[greater] - 2 * eta * _lambda

    return (weight_prime, bias_prime)


@problem.tag("hw2-A")
def loss(
        X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """

    return (np.sum((X @ weight + bias - y) ** 2) + _lambda * np.sum(np.abs(weight)))


@problem.tag("hw2-A", start_line=5)
def train(
        X: np.ndarray,
        y: np.ndarray,
        _lambda: float = 0.01,
        eta: float = 1e-5,
        convergence_delta: float = 1e-4,
        start_weight: np.ndarray = None,
        start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: Optional[np.ndarray] = None

    [weight, bias] = [start_weight, start_bias]

    convergence = False
    old_loss = current_loss = loss(X, y, weight, bias, _lambda)

    while convergence == False:
        old_loss = current_loss
        old_w = np.copy(weight)
        old_b = np.copy(bias)
        weight, bias = step(X, y, weight, bias, _lambda, eta)
        current_loss = loss(X, y, weight, bias, _lambda)
        if current_loss > old_loss:
            print('Ya dumb') #debugging
        convergence = convergence_criterion(weight, old_w, bias, old_b, convergence_delta)

    return (weight, bias)


@problem.tag("hw2-A")
def convergence_criterion(
        weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    conv = np.max(np.abs(weight - old_w)) < convergence_delta
    return (conv)


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """

    n = 500
    d = 1000
    k = 100

    weight_true = np.array([(j + 1) / k for j in range(k)] + [0] * (d - k))

    epsi = np.random.normal(0, 1, size=n)

    X_dumby = np.random.logistic(0, 2, (n, d))
    mu = np.mean(X_dumby, axis=0)
    std = np.std(X_dumby, axis=0)
    X = (X_dumby - mu) / std

    y = X @ weight_true + epsi

    _lambda_max = 2 * np.max(np.abs(X.T @ (y - np.mean(y))))
    _lambda = []
    weight_hat = []
    bias = []
    count = []
    FDR = []
    TPR = []

    for i in range(20):
        _lambda.append(_lambda_max / (2 ** i))
        weight_train, bias_train = train(X, y, _lambda[i])
        weight_hat.append(weight_train)
        bias.append(bias_train)
        count.append(d - np.sum(np.isclose(weight_hat[i], 0)))
        FDR.append(((d - k) - np.sum(np.isclose(weight_hat[i][k:], 0))) / count[i] if count[i] != 0 else 0)
        TPR.append(((k - np.sum(np.isclose(weight_hat[i][:k], 0))) / k))
        if count[i] == d:
            break

    # plotting the nonzeros and lambda values
    plt.subplot(2, 1, 1)
    plt.plot(_lambda, count)
    plt.xscale('log')
    plt.xlabel("Lambda Values")
    plt.ylabel("Nonzeros")
    plt.title(f"Regularization of Weights")
    # plotting the ROC
    plt.subplot(2, 1, 2)
    plt.plot(FDR, TPR)
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.savefig('question5.pdf')
    plt.show()


if __name__ == "__main__":
    main()
