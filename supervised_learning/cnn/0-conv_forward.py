#!/usr/bin/env python3

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        activation: activation function
        padding: "same" or "valid"
        stride: tuple (sh, sw)

    Returns:
        The output of the convolutional layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calculate output size and padding
    if padding == "same":
        h_new = int(np.ceil(h_prev / sh))
        w_new = int(np.ceil(w_prev / sw))

        pad_h = max((h_new - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_new - 1) * sw + kw - w_prev, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

    elif padding == "valid":
        h_new = int((h_prev - kh) / sh) + 1
        w_new = int((w_prev - kw) / sw) + 1

        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0

    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Apply padding
    A_pad = np.pad(
        A_prev,
        ((0, 0),
         (pad_top, pad_bottom),
         (pad_left, pad_right),
         (0, 0)),
        mode="constant"
    )

    # Initialize output
    Z = np.zeros((m, h_new, w_new, c_new))

    # Convolution
    for i in range(h_new):
        for j in range(w_new):

            h_start = i * sh
            h_end = h_start + kh

            w_start = j * sw
            w_end = w_start + kw

            window = A_pad[:, h_start:h_end, w_start:w_end, :]

            for c in range(c_new):
                Z[:, i, j, c] = (
                    np.sum(window * W[:, :, :, c], axis=(1, 2, 3))
                    / (kh * kw * c_prev)
                    + b[0, 0, 0, c]
                )

    # Apply activation
    A = activation(Z)

    return A
