#!/usr/bin/env python3
import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a neural network."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    A_prev_padded = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                           mode='constant', constant_values=0)

    h_new = int(((h_prev + 2 * ph - kh) / sh) + 1)
    w_new = int(((w_prev + 2 * pw - kw) / sw) + 1)

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for f in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw
                a_slice = A_prev_padded[:, vert_start:vert_end, horiz_start:horiz_end, :]
                Z[:, i, j, f] = np.sum(a_slice * W[:, :, :, f], axis=(1, 2, 3)) + b[0, 0, 0, f]

    A = activation(Z)
    return A
