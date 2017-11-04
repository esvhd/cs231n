import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        #######################################################################
        # TODO: Initialize the weights and biases of the two-layer net.
        # Weights    #
        # should be initialized from a Gaussian with standard deviation equal
        # to   #
        # weight_scale, and biases should be initialized to zero. All weights
        # and  #
        # biases should be stored in the dictionary self.params, with first
        # layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer
        # weights #
        # and biases using the keys 'W2' and 'b2'.              #
        #######################################################################
        W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
        b2 = np.zeros(num_classes)

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        self.output_size = num_classes
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.reg = reg
        #######################################################################
        #                             END OF YOUR CODE                      #
        #######################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter names to gradients of the loss with respect to those
        parameters.
        """
        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the two-layer net, computing
        # the    #
        # class scores for X and storing them in the scores
        # variable.              #
        #######################################################################
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        N = X.shape[0]
        X_reshaped = X.reshape(N, -1)

        z1 = np.dot(X_reshaped, W1) + b1
        h1, h1_cache = relu_forward(z1)

        scores = np.dot(h1, W2) + b2
        probs = softmax(scores, axis=1)
        #######################################################################
        #                             END OF YOUR CODE                     #
        #######################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the
        # loss  #
        # in the loss variable and gradients in the grads dictionary. Compute
        # data #
        # loss using softmax, and make sure that grads[k] holds the gradients
        # for  #
        # self.params[k]. Don't forget to add L2 regularization!         #
        #                                                                   #
        # NOTE: To ensure that your implementation matches ours and you pass
        # the   #
        # automated tests, make sure that your L2 regularization includes a
        # factor #
        # of 0.5 to simplify the expression for the
        # gradient.                      #
        #######################################################################
        logprobs = np.log(probs)
        loss += -np.sum(logprobs[range(N), y])
        loss /= N
        loss += .5 * self.reg * np.sum(np.multiply(W1, W1))
        loss += .5 * self.reg * np.sum(np.multiply(W2, W2))

        # backprop
        delta = np.zeros_like(probs)
        delta[range(N), y] = 1
        dz2 = probs - delta
        assert(dz2.shape == (N, self.output_size))

        dW2 = h1.T.dot(dz2) / N
        assert(dW2.shape == W2.shape)
        # bias = sum of gradients from all examples
        db2 = np.sum(dz2, axis=0) / N
        assert(db2.shape == (self.output_size,))

        dr1 = dz2.dot(W2.T)
        assert(dr1.shape == (N, self.hidden_size))
        dz1 = relu_backward(dr1, h1_cache)

        dW1 = X_reshaped.T.dot(dz1) / N
        assert(dW1.shape == W1.shape), '{} {}'.format(dW1.shape, W1.shape)
        db1 = np.sum(dz1, axis=0) / N

        # add regularization terms
        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        #######################################################################
        #                             END OF YOUR CODE                     #
        #######################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If
        dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch
        normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed
        using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
        This
          will make the dropout layers deteriminstic so we can gradient check
          the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all values
        # in    #
        # the self.params dictionary. Store weights and biases for the first
        # layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights
        # should be #
        # initialized from a normal distribution with standard deviation equal
        # to  #
        # weight_scale and biases should be initialized to zero.            #
        #                                                                  #
        # When using batch normalization, store scale and shift parameters for
        # the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and#
        # beta2, etc. Scale parameters should be initialized to one and shift #
        # parameters should be initialized to zero.                          #
        #######################################################################
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.output_dim = num_classes

        net_dim = [input_dim, hidden_dims, output_dim]

        L = len(hidden_dims)
        # initialize weights, indexing starts from 1
        for i in range(1, self.num_layers + 1):
            if i < 2:
                D = input_dim
            else:
                D = hidden_dims[i - 2]

            if i < L + 1:
                M = hidden_dims[i - 1]
            else:
                M = self.output_dim

            # print(i, (D, M))
            self.params['W' + str(i)] = np.random.randn(D, M) * weight_scale
            self.params['b' + str(i)] = np.zeros(M)

            if self.use_batchnorm and (i < self.num_layers) and i > 1:
                # TODO here, need to make sure indexing works.
                self.params['gamma' + str(i)] = np.ones(net_dim[i])
                self.params['beta' + str(i)] = np.zeros(net_dim[i])
        #######################################################################
        #                             END OF YOUR CODE                   #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and
        # the mode
        # (train / test). You can pass the same dropout_param to each dropout
        # layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the
        # forward pass
        # of the first batch normalization layer, self.bn_params[1] to the
        # forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the fully-connected net,
        # computing  #
        # the class scores for X and storing them in the scores variable.     #
        #                                                                   #
        # When using dropout, you'll need to pass self.dropout_param to each #
        # dropout forward pass.                                               #
        #                                                                     #
        # When using batch normalization, you'll need to pass
        # self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass      #
        # self.bn_params[1] to the forward pass for the second batch
        # normalization #
        # layer, etc.                                                    #
        #######################################################################
        L = len(self.hidden_dims)
        num_train = X.shape[0]
        h_prev = X.reshape(num_train, -1)
        W_sums = 0
        affine_cache = {}
        act_cache = {}
        dropout_cache = {}
        # bn_cache = {}

        for i in range(1, self.num_layers):
            W = self.params.get('W' + str(i))
            b = self.params.get('b' + str(i))
            assert(W is not None)
            if i < L + 1:
                assert(W.shape == (h_prev.shape[1], self.hidden_dims[i - 1]))
            else:
                assert(W.shape == (h_prev.shape[1], self.output_dim))
            assert(b is not None)

            # batchnorm
            if self.use_batchnorm:
                h_prev, h_cache = affine_batchnorm_relu_forward(x, W, b)
            else:
                z, a_cache = affine_forward(h_prev, W, b)
                h_prev, h_cache = relu_forward(z)

            # dropout
            if self.use_dropout:
                h_prev, u_cache = dropout_forward(h_prev, self.dropout_param)
                dropout_cache[i] = u_cache

            # accumulate for regularization
            W_sums += np.sum(W * W)

            affine_cache[i] = a_cache
            act_cache[i] = h_cache

        # last layer, softmax head
        W = self.params.get('W' + str(L + 1))
        b = self.params.get('b' + str(L + 1))

        scores, scores_cache = affine_forward(h_prev, W, b)
        W_sums += np.sum(W * W)
        #######################################################################
        #                             END OF YOUR CODE                      #
        #######################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #######################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store
        # the #
        # loss in the loss variable and gradients in the grads dictionary.
        # Compute #
        # data loss using softmax, and make sure that grads[k] holds the
        # gradients #
        # for self.params[k]. Don't forget to add L2 regularization!          #
        #                                                                   #
        # When using batch normalization, you don't need to regularize the
        # scale   #
        # and shift parameters.                                               #
        #                                                                     #
        # NOTE: To ensure that your implementation matches ours and you pass
        # the   #
        # automated tests, make sure that your L2 regularization includes a
        # factor #
        # of 0.5 to simplify the expression for the gradient.              #
        #######################################################################
        probs = softmax(scores, axis=1)
        logprobs = np.log(probs + 1e-8)
        loss += -np.sum(logprobs[range(logprobs.shape[0]), y])
        loss /= num_train

        # regularization
        loss += .5 * self.reg * W_sums

        # backprop through softmax and last affine layer
        delta = np.zeros_like(scores)
        delta[range(scores.shape[0]), y] = 1
        dscore = probs - delta
        # need to average over all examples. This would ensure 1/N is carried
        # back to the weights through the chain rule.
        dscore /= num_train

        # loss2, dscore2 = softmax_loss(scores, y)
        # assert(np.allclose(dscore, dscore2))

        dxL, dWL, dbL = affine_backward(dscore, scores_cache)

        # test averaging at the end
        # following has the same effect, but the averaging takes place
        # at the end of the chain.
        # dxx, dwx, dbx = affine_backward(dscore * num_train, scores_cache)
        # dwx /= num_train
        # doutx = dxx
        # assert(np.allclose(dwx, dWL))

        # whether to get weights from cache or params are the same.
        _, W, _ = scores_cache
        W2 = self.params['W' + str(L + 1)]
        assert(np.allclose(W, W2))
        grads['W' + str(L + 1)] = dWL + self.reg * W
        grads['b' + str(L + 1)] = dbL

        # through all other layers
        dout = dxL
        for ii in reversed(range(1, self.num_layers)):
            # print(ii)
            if self.use_dropout:
                u_cache = dropout_cache.get(ii)
                dout = dropout_backward(dout, u_cache)

            x = act_cache.get(ii)
            assert(x is not None)
            drelu = relu_backward(dout, x)

            a_cache = affine_cache.get(ii)
            assert(a_cache is not None)
            dout, dWL, dbL = affine_backward(drelu, a_cache)
            # dout is passed to the next rele_backward

            # accumulate gradient
            _, W, _ = a_cache
            W2 = self.params['W' + str(ii)]
            assert(np.allclose(W, W2))

            # doing averaging here failed grad_checks, but it should be
            # the same. Can write a test case like the one used for the last
            # affine layer to validate.
            # dW = dWL / num_train
            # db = dbL / num_train

            # test case below, it actually passes.
            # drelux = relu_backward(doutx, x)
            # doutx, dwx, dbx = affine_backward(drelux, a_cache)
            # dwx /= num_train
            # assert(np.allclose(dwx, dWL))

            grads['W' + str(ii)] = dWL + self.reg * W
            grads['b' + str(ii)] = dbL
        #######################################################################
        #                             END OF YOUR CODE                     #
        #######################################################################

        return loss, grads


def softmax(z, i=None, axis=None, eps=1e-8):
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
    prob = top / (bottom + eps)
    if i is not None:
        assert(prob.shape == (z.shape[0],))
    else:
        assert(prob.shape == z.shape)
    return prob
