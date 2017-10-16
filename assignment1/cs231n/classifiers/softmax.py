import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train, D = X.shape

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not carefu   #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                                        #
    ##########################################################################
    scores = X.dot(W)
    # socre.shape (N, C)
    z = softmax(scores, i=None, axis=1)
    logprob = np.log(z)
    # logprob.shape N x C
    loss += -np.sum(logprob[range(len(logprob)), y])

    loss /= num_train
    loss += .5 * reg * np.sum(np.multiply(W, W))

    correct_probs = logprob[range(num_train), y]
    assert(correct_probs.shape == (num_train,))

    # see DL notes for backprop derivatives.
    delta = np.zeros_like(scores)
    delta[range(num_train), y] = 1
    dscore = z - delta
    # dW.sahpe == D x C
    dW += X.T.dot(dscore)
    dW /= num_train
    dW += reg * W

    ##########################################################################
    #                          END OF YOUR CODE                              #
    ##########################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train, D = X.shape

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.#
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                                         #
    ##########################################################################
    scores = X.dot(W)
    # socre.shape (N, C)
    z = softmax(scores, i=None, axis=1)
    logprob = np.log(z)
    # logprob.shape N x C
    loss += -np.sum(logprob[range(len(logprob)), y])

    loss /= num_train
    loss += .5 * reg * np.sum(np.multiply(W, W))

    correct_probs = logprob[range(num_train), y]
    assert(correct_probs.shape == (num_train,))

    # see DL notes for backprop derivatives.
    delta = np.zeros_like(scores)
    delta[range(num_train), y] = 1
    dscore = z - delta
    # dW.sahpe == D x C
    dW += X.T.dot(dscore)
    dW /= num_train
    dW += reg * W

    ##########################################################################
    #                          END OF YOUR CODE                               #
    ##########################################################################

    return loss, dW


def softmax(z, i=None, axis=None):
    '''
    Numerically stable softmax.

    Args:
        z (TYPE): input data
        i (None, optional): index of class label(s) whose probabilities would
        be calculated.
        axis (None, optional): 1 if normalizing by rows, 0 if by columns

    Returns:
        (numpy.ndarray) Normalized softmax probabilities.
    '''
    max_z = np.max(z, axis=axis)
    z_mod = z - max_z.reshape(-1, 1)
    if i is not None:
        if z_mod.ndim < 2:
            top = np.exp(z_mod[i])
        else:
            # for every row, select based on i
            top = np.exp(z_mod[range(len(z_mod)), i])
    else:
        top = np.exp(z_mod)
    bottom = np.sum(np.exp(z_mod), axis=axis, keepdims=True)
    prob = top / bottom
    if i is not None:
        assert(prob.shape == (z.shape[0],))
    else:
        assert(prob.shape == z.shape)
    return prob
