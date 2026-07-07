#!/usr/bin/env python3
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions
    h_new = int(1 + (h_prev - kh) / sh)
    w_new = int(1 + (w_prev - kw) / sw)

    # Initialize output
    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            a_prev_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(a_prev_slice, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return A
