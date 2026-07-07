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
        numpy.ndarray containing the output of the convolution
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calculate padding
    if padding == "same":
        h_out = int(np.ceil(h_prev / sh))
        w_out = int(np.ceil(w_prev / sw))

        pad_h = max((h_out - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_out - 1) * sw + kw - w_prev, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

    elif padding == "valid":
        pad_top = pad_bottom = 0
        pad_left = pad_right = 0

        h_out = int((h_prev - kh) / sh) + 1
        w_out = int((w_prev - kw) / sw) + 1

    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Apply padding
    A_pad = np.pad(
        A_prev,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
    )

    # Initialize output
    Z = np.zeros((m, h_out, w_out, c_new))

    # Convolution
    for i in range(h_out):
        for j in range(w_out):
            vert_start = i * sh
            vert_end = vert_start + kh

            horiz_start = j * sw
            horiz_end = horiz_start + kw

            a_slice = A_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]

            for c in range(c_new):
                Z[:, i, j, c] = (
                    np.sum(a_slice * W[:, :, :, c], axis=(1, 2, 3)) + b[:, :, :, c]
                )

    # Apply activation
    A = activation(Z)

    return A
