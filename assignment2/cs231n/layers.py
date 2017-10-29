import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ##########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ##########################################################################
    out = np.dot(x.reshape(x.shape[0], -1), w) + b
    ##########################################################################
    #                             END OF YOUR CODE                          #
    ##########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ##########################################################################
    # TODO: Implement the affine backward pass.                              #
    ##########################################################################
    # (N, M) * (M, D)
    dx = np.dot(dout, w.T).reshape(x.shape)
    # (D, N) * (N, M)
    dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
    # sum across all rows of dout
    db = np.sum(dout, axis=0)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ##########################################################################
    # TODO: Implement the ReLU forward pass.                                #
    ##########################################################################
    out = np.maximum(0, x)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ##########################################################################
    # TODO: Implement the ReLU backward pass.                                #
    ##########################################################################
    dx = np.zeros_like(x)
    dx[x > 0] = 1
    dx *= dout
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to
    normalize data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch
        # normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these#
        # statistics to normalize the incoming data, and scale and shift the#
        # normalized data using gamma and beta.                               #
        #                                                                    #
        # You should store the output in the variable out. Any intermediates
        # that   #
        # you need for the backward pass should be stored in the cache
        # variable.    #
        #                                                                    #
        # You should also use your computed sample mean and variance together
        # with  #
        # the momentum variable to update the running mean and running
        # variance,    #
        # storing your result in the running_mean and running_var
        # variables.        #
        #######################################################################
        mean = np.mean(x, axis=0)
        # use population variance rather than sample mean here. hmm..
        variance = np.var(x, axis=0, ddof=0)
        x_norm = (x - mean) / np.sqrt(variance + eps)

        running_mean *= momentum
        running_mean += (1 - momentum) * mean
        running_var *= momentum
        running_var += (1 - momentum) * variance

        out = gamma * x_norm + beta

        cache = (x, mean, variance, x_norm, gamma, eps)
        #######################################################################
        #                             END OF YOUR CODE            #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization.
        # Use   #
        # the running mean and variance to normalize the incoming data, then
        # scale  #
        # and shift the normalized data using gamma and beta. Store the result
        # in   #
        # the out variable.                                                #
        #######################################################################
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        #######################################################################
        #                             END OF YOUR CODE                 #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the   #
    # results in the dx, dgamma, and dbeta variables.                       #
    ##########################################################################
    x, mean, var, x_norm, gamma, eps = cache
    N, D = dout.shape

    xmu = x - mean
    var_eps = var + eps

    dbeta = np.sum(dout, axis=0)
    assert(dbeta.shape == (D,)), dbeta.shape

    # expects (D,)
    dgamma = np.sum(x_norm * dout, axis=0)
    assert(dgamma.shape == (D,)), dgamma.shape

    # N x D
    dxn = dout * gamma
    assert(dxn.shape == (N, D))

    # D x 1
    # let:
    # f(x) = x^{-1/2} -> f' = -1/2 * x^{3/2}
    # h(x, mu) = x - mu
    # g(x, mu, f(v)) = h(x, mu) * f(v)
    # shape: (D,) * [(N, D) - (D,)] * (D,) = N, D
    # then summed over N
    # g'_{f(v)} = (x - mu)
    dvar = np.sum(dxn * xmu, axis=0) * -1 / 2 * np.power(var_eps, -3 / 2)
    assert(dvar.shape == (D,)), dvar.shape

    # (N, D)
    # g'_{x, mu} = f(v)
    d_xmu = dxn * 1 / np.sqrt(var_eps)
    assert(d_xmu.shape == (N, D))

    # (N, D) * (D, 1) = (N, D)
    d_xmusq = np.ones_like(x) / N * dvar
    assert(d_xmusq.shape == (N, D)), d_xmusq.shape

    # N,D
    dxmu2 = 2 * xmu * d_xmusq
    assert(dxmu2.shape == (N, D))

    # N x D
    dx1 = d_xmu + dxmu2
    assert(dx1.shape == (N, D)), dx.shape

    # (D,)
    dmu = -np.sum(d_xmu + dxmu2, axis=0)
    assert(dmu.shape == (D,))

    # (N, D)
    dx2 = np.ones_like(x) / N * dmu

    # (N, D)
    dx = dx1 + dx2
    ##########################################################################
    #                             END OF YOUR CODE                          #
    ##########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the   #
    # results in the dx, dgamma, and dbeta variables.                     #
    #                                                                        #
    # After computing the gradient with respect to the centered inputs, you #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ##########################################################################
    x, mean, var, x_norm, gamma, eps = cache
    N, D = dout.shape

    mu = 1 / N * np.sum(x, axis=0)
    xmu = x - mu

    var = 1 / N * np.sum((x - mu)**2, axis=0)
    var_eps = var + eps

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm * dout, axis=0)
    # credit to this blog post
    # http://cthorey.github.io./backpropagation/
    dx = (1 / N * gamma * np.power(var_eps, -.5) *
          (N * dout - np.sum(dout, axis=0) -
           xmu / var_eps * np.sum(dout * xmu, axis=0)))
    ##########################################################################
    #                             END OF YOUR CODE                          #
    ##########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the
    dropout mask that was used to multiply the input; in test mode,
    mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement the training phase forward pass for inverted
        # dropout.   #
        # Store the dropout mask in the mask variable.  #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = np.multiply(x, mask)
        #######################################################################
        #                            END OF YOUR CODE                #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout. #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training phase backward pass for inverted
        # dropout.  #
        #######################################################################
        dx = np.multiply(dout, mask)
        #######################################################################
        #                            END OF YOUR CODE               #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ##########################################################################
    # TODO: Implement the convolutional forward pass.            #
    # Hint: you can use the function np.pad for padding.                 #
    ##########################################################################
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    assert(H >= HH and W >= WW), 'Filter larger than input!'
    stride, pad = conv_param.get('stride'), conv_param.get('pad')
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # prepare output
    out = np.zeros((N, F, H_out, W_out))

    # pad only along H and W
    # pad_width here is a tuple of tuples, each indicating the pad along
    # an axis, (pad,) is short for (pad,pad)
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                   'constant', constant_values=0)

    for n in range(N):
        # for each example
        for f in range(F):
            # for each filter
            for ho in range(H_out):
                for wo in range(W_out):
                    # for each cell in output
                    hx = ho * stride
                    wx = wo * stride
                    # filter is applied to the full depth of the region
                    xx = x_pad[n, :, hx:hx + HH, wx:wx + WW]
                    out[n, f, ho, wo] = np.sum(xx * w[f]) + b[f]
    ##########################################################################
    #                             END OF YOUR CODE                     #
    ##########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ##########################################################################
    # TODO: Implement the convolutional backward pass.                 #
    ##########################################################################
    x, w, b, conv_param = cache
    stride, pad = conv_param.get('stride'), conv_param.get('pad')
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    # dout.shape == (N, F, H_out, W_out)

    # db.shape = (F,)
    db = dout.sum(axis=3).sum(axis=2).sum(axis=0)
    assert(db.shape == b.shape)

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                   'constant', constant_values=0)

    # dw.shape == (F, C, HH, WW)
    dw = np.zeros_like(w)
    # dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)

    for n in range(N):
        for f in range(F):
            for ho in range(H_out):
                for wo in range(W_out):
                    hh = ho * stride
                    ww = wo * stride

                    m = x_pad[n, :, hh:hh + HH, ww:ww + WW]
                    assert(m.shape == (C, HH, WW))
                    # sum over all images, all local regions matching filter
                    # size
                    dw[f] += m * dout[n, f, ho, wo]

                    # sum over all filters
                    dx_pad[n, :,
                           hh:hh + HH,
                           ww:ww + WW] += w[f] * dout[n, f, ho, wo]
    # unpad
    dx = dx_pad[:, :, 1:-1, 1:-1]
    assert(dx.shape == x.shape)
    ##########################################################################
    #                             END OF YOUR CODE               #
    ##########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ##########################################################################
    # TODO: Implement the max pooling forward pass               #
    ##########################################################################
    pw = pool_param.get('pool_width')
    ph = pool_param.get('pool_height')
    stride = pool_param.get('stride')

    N, C, H, W = x.shape

    H_out = 1 + (H - ph) // stride
    W_out = 1 + (W - pw) // stride

    out = np.zeros((N, C, H_out, W_out))

    # local gradient
    dx_local = np.zeros_like(x)

    for n in range(N):
        for hh in range(H_out):
            for ww in range(W_out):
                h_ind = hh * stride
                w_ind = ww * stride
                area = x[n, :, h_ind:h_ind + ph, w_ind:w_ind + pw]
                assert(area.shape == (C, ph, pw))
                pooled = np.max(np.max(area, axis=2), axis=1)
                assert(pooled.shape == (C,)), pooled.shape
                out[n, :, hh, ww] = pooled

                # gradient is essentially 1 for the max value and 0 elsewhere
                grad_area = dx_local[n, :, h_ind:h_ind + ph, w_ind:w_ind + pw]
                for c in range(C):
                    flag = np.isclose(area[c], pooled[c])
                    assert(flag.shape == (ph, pw))
                    grad_area[c][flag] += 1

    pool_param['grad_local'] = dx_local
    ##########################################################################
    #                             END OF YOUR CODE                #
    ##########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ##########################################################################
    # TODO: Implement the max pooling backward pass              #
    ##########################################################################
    x, pool_param = cache
    pw = pool_param.get('pool_width')
    ph = pool_param.get('pool_height')
    stride = pool_param.get('stride')
    N, C, H, W = x.shape

    # fast way below but will only have values if forward pass is called
    # first.
    grad_local = pool_param.get('grad_local')
    assert(grad_local is not None), 'Forward pass not done? '
    'Missing local gradient'

    N, C, H_out, W_out = dout.shape
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for hh in range(H_out):
                for ww in range(W_out):
                    h_ind = hh * stride
                    w_ind = ww * stride
                    dx_area = dx[n, c, h_ind:h_ind + ph, w_ind:w_ind + pw]
                    loc_area = grad_local[n, c,
                                          h_ind:h_ind + ph,
                                          w_ind:w_ind + pw]
                    dx_area += loc_area * dout[n, c, hh, ww]

    ##########################################################################
    #                             END OF YOUR CODE                  #
    ##########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ##########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.     #
    #                                                                       #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                 #
    ##########################################################################
    N, C, H, W = x.shape

    # key here is that the normalization happens along 3 out of 4 axis of x,
    # all images for a given channel.
    # we can simply rearrange x to make it fit into a 2D format.
    x = x.swapaxes(0, 1)
    out, cache = batchnorm_forward(x.reshape((C, -1)).T,
                                   gamma, beta, bn_param)
    out = out.T.reshape((C, N, H, W)).swapaxes(0, 1)

    # below is a manual implementation
    # mode = bn_param.get('mode')
    # assert(mode == 'train' or mode == 'test'), 'Invalid BatchNorm mode.'

    # eps = bn_param.get('eps', 1e-8)
    # momentum = bn_param.get('momentum', 0.9)
    # running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    # running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

    # if mode == 'train':
    #     mean = x.mean(axis=(0, 2, 3))
    #     var = x.var(axis=(0, 2, 3), ddof=1)

    #     x_norm = (x - mean.reshape((1, C, 1, 1))) / \
    #         np.sqrt(var.reshape((1, C, 1, 1)) + eps)
    #     assert(x_norm.shape == x.shape), (x_norm.shape, x.shape)

    #     out = gamma.reshape((1, C, 1, 1)) * x_norm + beta.reshape((1, C, 1, 1))

    #     running_mean *= momentum
    #     running_mean += (1 - momentum) * mean

    #     running_var *= momentum
    #     running_var += (1 - momentum) * var

    #     bn_param['running_mean'] = running_mean
    #     bn_param['running_var'] = running_var

    #     cache = (x, mean, var, x_norm, gamma, eps)
    # elif mode == 'test':
    #     x_norm = (x - running_mean.reshape((1, C, 1, 1))) / \
    #         np.sqrt(running_var.reshape((1, C, 1, 1)) + eps)
    #     out = gamma.reshape((1, C, 1, 1)) * x_norm + beta.reshape((1, C, 1, 1))
    # else:
    #     pass
    ##########################################################################
    #                             END OF YOUR CODE                 #
    ##########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ##########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.  #
    #                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla  #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                    #
    ##########################################################################
    N, C, H, W = dout.shape

    # treat each channel as an example
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.swapaxes(0, 1)
                                               .reshape((C, -1)).T,
                                               cache)
    dx = dx.T.reshape(C, N, H, W).swapaxes(0, 1)

    ##########################################################################
    #                             END OF YOUR CODE                    #
    ##########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
