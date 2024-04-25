#!/usr/bin/env python

#######################################################
# Copyright (c) 2024, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import sys
import time

from mnist_common import display_results, setup_mnist

import arrayfire as af


def accuracy(predicted: af.Array, target: af.Array) -> float | complex:
    """Calculates the accuracy of the predictions compares to the actual target"""
    _, tlabels = af.imax(target, axis=1)
    _, plabels = af.imax(predicted, axis=1)
    return 100 * af.count(plabels == tlabels) / tlabels.size


def abserr(predicted: af.Array, target: af.Array) -> float | complex:
    """Calculates the mean absolute error (MAE), scaled by 100"""
    return 100 * af.sum(af.abs(predicted - target)) / predicted.size


def predict_prob(X: af.Array, Weights: af.Array) -> af.Array:
    """Predict (probability) based on given parameters"""
    Z = af.matmul(X, Weights)
    return af.sigmoid(Z)


def predict_log_prob(X: af.Array, Weights: af.Array) -> af.Array:
    """# Predict (log probability) based on given parameters"""
    return af.log(predict_prob(X, Weights))


def predict_class(X: af.Array, Weights: af.Array) -> af.Array:
    """Give most likely class based on given parameters"""
    probs = predict_prob(X, Weights)
    _, classes = af.imax(probs, axis=1)
    return classes


def cost(Weights: af.Array, X: af.Array, Y: af.Array, lambda_param: float = 1.0) -> tuple[af.Array, af.Array]:
    """Calculate the cost of predictions made with a given set of weights"""
    # Number of samples
    m = Y.shape[0]

    dim0 = Weights.shape[0]
    dim1 = Weights.shape[1] if len(Weights.shape) > 1 else 1
    dim2 = Weights.shape[2] if len(Weights.shape) > 2 else 1
    dim3 = Weights.shape[3] if len(Weights.shape) > 3 else 1
    # Make the lambda corresponding to Weights(0) == 0
    lambdat = af.constant(lambda_param, (dim0, dim1, dim2, dim3))

    # No regularization for bias weights
    lambdat[0, :] = 0

    # Get the prediction
    H = predict_prob(X, Weights)

    # Cost of misprediction
    Jerr = -1 * af.sum(Y * af.log(H) + (1 - Y) * af.log(1 - H), axis=0)

    # Regularization cost
    Jreg = 0.5 * af.sum(lambdat * Weights * Weights, axis=0)

    # Total cost
    J = (Jerr + Jreg) / m

    # Find the gradient of cost
    D = H - Y
    dJ = (af.matmul(X, D, af.MatProp.TRANS) + lambdat * Weights) / m

    return J, dJ


def train(
    X: af.Array,
    Y: af.Array,
    alpha: float = 0.1,
    lambda_param: float = 1.0,
    maxerr: float = 0.01,
    maxiter: int = 1000,
    verbose: bool = False,
) -> af.Array:  # noqa :E501
    """Train a machine learning model using gradient descent to minimize the cost function."""
    # Initialize parameters to 0
    Weights = af.constant(0, (X.shape[1], Y.shape[1]))

    for i in range(maxiter):
        # Get the cost and gradient
        J, dJ = cost(Weights, X, Y, lambda_param)

        err = af.max(af.abs(J))
        if err < maxerr:  # type: ignore[operator]
            print("Iteration {0:4d} Err: {1:4f}".format(i + 1, err))  # type: ignore[str-format]
            print("Training converged")
            return Weights

        if verbose and ((i + 1) % 10 == 0):
            print("Iteration {0:4d} Err: {1:4f}".format(i + 1, err))  # type: ignore[str-format]

        # Update the parameters via gradient descent
        Weights = Weights - alpha * dJ

    if verbose:
        print("Training stopped after {0:d} iterations".format(maxiter))

    return Weights


def benchmark_logistic_regression(train_feats: af.Array, train_targets: af.Array, test_feats: af.Array) -> None:
    t0 = time.time()
    Weights = train(train_feats, train_targets, 0.1, 1.0, 0.01, 1000)
    af.eval(Weights)
    af.sync()
    t1 = time.time()
    dt = t1 - t0
    print("Training time: {0:4.4f} s".format(dt))

    t0 = time.time()
    iters = 100
    for i in range(iters):
        test_outputs = predict_prob(test_feats, Weights)
        af.eval(test_outputs)
    af.sync()
    t1 = time.time()
    dt = t1 - t0
    print("Prediction time: {0:4.4f} s".format(dt / iters))


def logit_demo(console: bool, perc: int) -> None:
    """Demo of one vs all logistic regression"""
    # Load mnist data
    frac = float(perc) / 100.0
    mnist_data = setup_mnist(frac, True)
    num_classes = mnist_data[0]  # noqa: F841
    num_train = mnist_data[1]
    num_test = mnist_data[2]
    train_images = mnist_data[3]
    test_images = mnist_data[4]
    train_targets = mnist_data[5]
    test_targets = mnist_data[6]

    # Reshape images into feature vectors
    feature_length = int(train_images.size / num_train)
    train_feats = af.transpose(af.moddims(train_images, (feature_length, num_train)))

    test_feats = af.transpose(af.moddims(test_images, (feature_length, num_test)))

    train_targets = af.transpose(train_targets)
    test_targets = af.transpose(test_targets)

    num_train = train_feats.shape[0]
    num_test = test_feats.shape[0]

    # Add a bias that is always 1
    train_bias = af.constant(1, (num_train, 1))
    test_bias = af.constant(1, (num_test, 1))
    train_feats = af.join(1, train_bias, train_feats)
    test_feats = af.join(1, test_bias, test_feats)

    # Train logistic regression parameters
    Weights = train(
        train_feats,
        train_targets,
        0.1,  # learning rate
        1.0,  # regularization constant
        0.01,  # max error
        1000,  # max iters
        True,  # verbose mode
    )  # noqa: E124

    af.eval(Weights)
    af.sync()

    # Predict the results
    train_outputs = predict_prob(train_feats, Weights)
    test_outputs = predict_prob(test_feats, Weights)

    print("Accuracy on training data: {0:2.2f}".format(accuracy(train_outputs, train_targets)))  # type: ignore[str-format] # noqa :E501
    print("Accuracy on testing data: {0:2.2f}".format(accuracy(test_outputs, test_targets)))  # type: ignore[str-format] # noqa :E501
    print("Maximum error on testing data: {0:2.2f}".format(abserr(test_outputs, test_targets)))  # type: ignore[str-format] # noqa :E501

    benchmark_logistic_regression(train_feats, train_targets, test_feats)

    if not console:
        test_outputs = af.transpose(test_outputs)
        # Get 20 random test images
        display_results(test_images, test_outputs, af.transpose(test_targets), 20, True)


def main() -> None:
    argc = len(sys.argv)

    device = int(sys.argv[1]) if argc > 1 else 0
    console = sys.argv[2][0] == "-" if argc > 2 else False
    perc = int(sys.argv[3]) if argc > 3 else 60

    try:
        af.set_device(device)
        af.info()
        logit_demo(console, perc)
    except Exception as e:
        print("Error: ", str(e))


if __name__ == "__main__":
    main()
