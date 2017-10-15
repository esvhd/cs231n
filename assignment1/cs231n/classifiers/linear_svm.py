import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train, dim = X.shape

    dW = np.zeros(W.shape)  # initialize the gradient as zero

    loss = 0.0
    for i in range(num_train):
        # (1, D) x (D, C) = (1, C)
        scores = X[i].dot(W)

        # scores shape 1 x C
        assert(scores.shape == (num_classes,))
        # X[i].shape == (D,)
        assert(X[i].shape == (dim,))

        # y[i] is the true class label of training example i
        # scores[y[i]] is therefore the score for the correct class
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            # j is a class label, scores[j] is the socre for this class
            # therefore j and y[i] are both class labels
            # ignore the case where j equals y[i],
            # i.e. ignore the score for the correct class.
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                # gradients should push UP the score of the correct class
                # and push DOWN the scores of incorrect classes.
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    dW += reg * W

    # my questino here though, is why that the gradient loss isn't derived
    # from the loss function and then back to W.
    # This is beause the derivative of the loss function is like ReLu and
    # is either 1 (if x > 0) or 0 otherwise.

    ##########################################################################
    # TODO:                                                                  #
    # Compute the gradient of the loss function and store it dW.             #
    # Rather that first computing the loss and then computing the derivative,#
    # it may be simpler to compute the derivative at the same time that the  #
    # loss is being computed. As a result you may need to modify some of the #
    # code above to compute the gradient.                                    #
    ##########################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.

    Ref:

    https://github.com/bruceoutdoors/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py

    https://github.com/huyouare/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py

    https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
    """
    loss = 0.0
    # W - (D, C) - column j is weights for computing score for class j
    # X - (N, D)
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    dim, num_classes = W.shape
    num_train = X.shape[0]

    ##########################################################################
    # TODO:                                                                  #
    # Implement a vectorized version of the structured SVM loss, storing the #
    # result in loss.                                                        #
    ##########################################################################
    scores = np.dot(X, W)
    # scores.shape == (N, C)
    assert(scores.shape == (num_train, num_classes))

    correct_class_scores = scores[range(len(scores)), y].reshape(-1, 1)
    assert(correct_class_scores.shape == (num_train, 1))

    # broadcast, loss_per_class.shape == (N, C)
    # in doing this, the correct class will have a value of 1,
    # but we do not use this value in the total loss, so need to be corrected.
    loss_per_class = scores - correct_class_scores + 1
    # correct loss for the correct class to be 0
    loss_per_class[range(len(scores)), y] = 0
    # clip margins < 0
    loss_per_class = loss_per_class.clip(min=0)

    loss = np.sum(loss_per_class) / num_train
    loss += .5 * reg * np.sum(np.multiply(W, W))

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    ##########################################################################
    # TODO:                                                                  #
    # Implement a vectorized version of the gradient for the structured SVM  #
    # loss, storing the result in dW.                                        #
    #                                                                        #
    # Hint: Instead of computing the gradient from scratch, it may be easier #
    # to reuse some of the intermediate values that you used to compute the  #
    # loss.                                                                  #
    ##########################################################################
    # backprop
    # (D, N) x (N, C) = (D, C)
    # for j != y_i
    dmargin = (loss_per_class > 0).astype(float)
    # for j == y_i
    dmargin[range(len(dmargin)), y] -= np.sum(dmargin, axis=1)

    dW += np.dot(X.T, dmargin)
    dW /= num_train
    dW += reg * W
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return loss, dW
