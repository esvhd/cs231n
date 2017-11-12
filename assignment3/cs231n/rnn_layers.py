import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


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


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a
    tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we
    use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN.
    # Store the next  #
    # hidden state and any values you need for the backward pass in the
    # next_h   #
    # and cache variables respectively.                                       #
    ##########################################################################
    yx = np.dot(x, Wx)
    yh_prev = np.dot(prev_h, Wh)

    z = yx + yh_prev + b
    assert(z.shape == prev_h.shape)

    next_h = np.tanh(z)
    cache = (Wx, Wh, x, prev_h, z)
    ##########################################################################
    #                               END OF YOUR CODE                         #
    ##########################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (N, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.   #
    #                                                                         #
    # HINT: For the tanh function, you can compute the local derivative in
    # terms #
    # of the output value from tanh.                                          #
    ##########################################################################
    Wx, Wh, x, prev_h, z = cache

    D, H = Wx.shape
    N, D = x.shape

    # dz.shape == (N, H)
    dz = (1 - np.power(np.tanh(z), 2)) * dnext_h
    assert(dz.shape == (N, H))

    # dx.shape = (N, D)
    dx = np.dot(dz, Wx.T)
    assert(dx.shape == (N, D))

    # dprev_h.shape == (N, H)
    dprev_h = np.dot(dz, Wh.T)
    assert(dprev_h.shape == (N, H))

    db = np.sum(dz, axis=0)
    assert(db.shape == (H,))

    dWx = np.dot(x.T, dz)
    dWh = np.dot(prev_h.T, dz)

    ##########################################################################
    #                               END OF YOUR CODE                         #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After
    running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of#
    # input data. You should use the rnn_step_forward function that you
    # defined  #
    # above.                                                                 #
    ##########################################################################
    xx = x.swapaxes(0, 1)
    T, N, D = xx.shape
    _, H = Wx.shape

    h = np.zeros((T, N, H))
    cache = []

    for i in range(T):
        if i < 1:
            prev_ht = h0
        else:
            prev_ht = h[i - 1]

        xt = xx[i]

        h[i], cache_t = rnn_step_forward(xt, prev_ht, Wx, Wh, b)
        cache.append(cache_t)

    h = h.swapaxes(0, 1)
    ##########################################################################
    #                               END OF YOUR CODE                          #
    ##########################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of
    data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire   #
    # sequence of data. You should use the rnn_step_backward function that you#
    # defined above.                                                         #
    ##########################################################################
    N, T, H = dh.shape
    dh1 = dh.swapaxes(0, 1)

    Wx, Wh, x, prev_h, z = cache[-1]
    _, D = x.shape

    dx = np.zeros((T, N, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros(H)
    dprev_h = np.zeros_like(prev_h)

    for i in reversed(range(T)):
        cache_t = cache[i]

        # gradient for upstream h is combined together
        dnext_ht = dh1[i] + dprev_h

        dxt, dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_ht, cache_t)

        dWx += dWxt
        dWh += dWht
        db += dbt
        dx[i] += dxt

    dh0 += dprev_h
    dx = dx.swapaxes(0, 1)
    ##########################################################################
    #                               END OF YOUR CODE                          #
    ##########################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning
    each to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element
    idx of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##########################################################################
    # TODO: Implement the forward pass for word embeddings.                   #
    #                                                                        #
    # HINT: This should be very simple.                                       #
    ##########################################################################
    N, T = x.shape
    _, D = W.shape
    # out = np.zeros((N * T, D))
    out = W[x.reshape(1, -1)].reshape((N, T, D))
    cache = (x, W)
    ##########################################################################
    #                               END OF YOUR CODE                          #
    ##########################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##########################################################################
    # TODO: Implement the backward pass for word embeddings.                  #
    #                                                                         #
    # HINT: Look up the function np.add.at                                    #
    ##########################################################################
    x, W = cache
    N, T, D = dout.shape

    # essentially using numbers to represent words. Goal of training is to
    # figure out the best number to represent each word.

    dW = np.zeros_like(W)
    np.add.at(dW, x.reshape(1, -1), dout.reshape(N * T, D))
    ##########################################################################
    #                               END OF YOUR CODE                         #
    ##########################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we
    use a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    ##########################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.     #
    # You may want to use the numerically stable sigmoid implementation above.#
    ##########################################################################

    N, D = x.shape
    _, H = prev_h.shape

    # computes ifog in one pop
    z = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    assert(z.shape == (N, 4 * H))

    ui = z[:, :H]
    uf = z[:, H:2 * H]
    uo = z[:, 2 * H:3 * H]
    ug = z[:, 3 * H:]

    i = sigmoid(ui)
    f = sigmoid(uf)
    o = sigmoid(uo)
    g = np.tanh(ug)

    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    cache = (x, i, f, o, g,
             prev_h, prev_c, Wx, Wh, b, next_c)

    ##########################################################################
    #                               END OF YOUR CODE                         #
    ##########################################################################

    return next_h, next_c, cache


def dtanh(x):
    return (1 - np.tanh(x)**2)


def lstm_ifo_gates_backward(dout, u, x, prev_h):
    N, D = x.shape
    _, H = prev_h.shape

    du = dsigmoid(u) * dout
    assert(du.shape == (N, H))

    dWx = np.dot(x.T, du)
    assert(dWx.shape == (D, H))

    dWh = np.dot(prev_h.T, du)
    assert(dWh.shape == (H, H))

    db = np.sum(du, axis=0)
    assert(db.shape == (H,))

    return du, dWx, dWh, db


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    # dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.   #
    #                                                                        #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of#
    # the output value from the nonlinearity.                               #
    ##########################################################################
    x, i, f, o, g, \
        prev_h, prev_c, Wx, Wh, b, next_c = cache

    N, D = x.shape
    _, H = prev_h.shape

    # allocate memory for du first. Array memory in numpy must be continuous,
    # therefore concatenating separately allocated arrays is not memory
    # efficient.
    du = np.zeros((N, 4 * H))

    # dc = dnext_c + dnext_h * o * dtanh(prev_c)
    # dnext_c here below is what I got wrong.
    dnext_c += dnext_h * o * dtanh(next_c)

    dprev_c = f * dnext_c

    # compute di, df, do, dg
    di = dnext_c * g
    df = dnext_c * prev_c
    dg = dnext_c * i

    do = dnext_h * np.tanh(next_c)
    assert(do.shape == (N, H))

    # backprop for ifog gates
    # dsigmoid * dupstream
    du[:, :H] = i * (1 - i) * di
    du[:, H:2 * H] = f * (1 - f) * df
    du[:, 2 * H:3 * H] = o * (1 - o) * do
    # dug = dtanh(ug) * dg
    du[:, 3 * H:] = dg * (1 - g**2)

    # du = np.concatenate((dui, duf, duo, dug), axis=1)
    # assert(du.shape == (N, 4 * H))

    # computer dx, dprev_h, dWh, dWx, db
    dx = du.dot(Wx.T)
    assert(dx.shape == (N, D))

    dWx = x.T.dot(du)
    assert(dWx.shape == Wx.shape)

    dWh = prev_h.T.dot(du)
    assert(dWh.shape == (H, 4 * H))

    dprev_h = du.dot(Wh.T)
    assert(dprev_h.shape == prev_h.shape)

    db = np.sum(du, axis=0)
    assert(db.shape == (4 * H, ))

    # dWxg = x.T.dot(dug)
    # assert(dWxg.shape == (D, H))

    # dWhg = prev_h.T.dot(dug)
    # assert(dWhg.shape == (H, H))

    # dbg = np.sum(dug, axis=0)
    # assert(dbg.shape == (H,))

    # # backprop for i, f, o gate
    # duo, dWxo, dWho, dbo = lstm_ifo_gates_backward(do, uo, x, prev_h)
    # dui, dWxi, dWhi, dbi = lstm_ifo_gates_backward(di, ui, x, prev_h)
    # duf, dWxf, dWhf, dbf = lstm_ifo_gates_backward(df, uf, x, prev_h)

    # # compute dx
    # Wxi = Wx[:, :H]
    # Wxf = Wx[:, H:2 * H]
    # Wxo = Wx[:, 2 * H:3 * H]
    # Wxg = Wx[:, 3 * H:]

    # dx = np.dot(dui, Wxi.T) + np.dot(duf, Wxf.T) + np.dot(duo, Wxo.T) +\
    #     np.dot(dug, Wxg.T)
    # assert(dx.shape == x.shape)

    # # compute dh
    # Whi = Wh[:, :H]
    # Whf = Wh[:, H:2 * H]
    # Who = Wh[:, 2 * H:3 * H]
    # Whg = Wh[:, 3 * H:]

    # dprev_h = dui.dot(Whi.T) + duf.dot(Whf.T) + duo.dot(Who.T) + dug.dot(Whg.T)
    # assert(dprev_h.shape == prev_h.shape)

    # # combine for Wx, Wh, b
    # dWx = np.concatenate((dWxi, dWxf, dWxo, dWxg), axis=1)
    # dWh = np.concatenate((dWhi, dWhf, dWho, dWhg), axis=1)
    # db = np.concatenate((dbi, dbf, dbo, dbg))
    ##########################################################################
    #                               END OF YOUR CODE                         #
    ##########################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an
    input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After
    running the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.#
    # You should use the lstm_step_forward function that you just defined.   #
    ##########################################################################
    N, T, D = x.shape
    _, H = h0.shape

    h = np.zeros((N, T, H))
    h[:, 0, :] = h0

    cache = []

    prev_c = 0.
    for t in range(T):
        if t < 1:
            prev_h = h0
        else:
            prev_h = h[:, t - 1, :]

        next_h, prev_c, step_cache = lstm_step_forward(x[:, t, :],
                                                       prev_h, prev_c,
                                                       Wx, Wh, b)

        cache.append(step_cache)
        h[:, t, :] = next_h
    ##########################################################################
    #                               END OF YOUR CODE                         #
    ##########################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.#
    # You should use the lstm_step_backward function that you just defined.  #
    ##########################################################################
    N, T, H = dh.shape
    # data shape
    _, D = cache[0][0].shape

    dx = np.zeros((T, N, D))
    # dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros(4 * H)

    dprev_c = np.zeros((N, H))
    dprev_h = 0

    dh = dh.swapaxes(0, 1)

    for t in reversed(range(T)):
        # next_h gradient is summed over upstream and previous step
        dnext_h = dh[t] + dprev_h

        # print(dnext_h.shape)

        dxt, dprev_h, dprev_c, dWxt, dWht, dbt = \
            lstm_step_backward(dnext_h, dprev_c, cache[t])

        dx[t] += dxt
        dWx += dWxt
        dWh += dWht
        db += dbt

    dh0 = dprev_h
    dx = dx.swapaxes(0, 1)
    ##########################################################################
    #                               END OF YOUR CODE                         #
    ##########################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of
    D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives
    scores for all vocabulary elements at all timesteps, and y gives the
    indices of the ground-truth element at each timestep. We use a
    cross-entropy loss at each timestep, summing the loss over all timesteps
     and averaging across the minibatch.

    As an additional complication, we may want to ignore the model output at
    some timesteps, since sequences of different length may have been combined
    into a minibatch and padded with NULL tokens.
    The optional mask argument tells us which elements should contribute to
    the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the
    range 0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
