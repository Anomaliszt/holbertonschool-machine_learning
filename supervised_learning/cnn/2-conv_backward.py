#!/usr/bin/env python3
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a neural network."""
    
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, c_new = dZ.shape

    # Calculate padding (same as in forward)
    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Pad A_prev
    A_prev_padded = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                           mode='constant', constant_values=0)

    # Initialize gradients
    dA_prev = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Compute gradients
    for i in range(m):
        for h in range(h_new):
            for w_pos in range(w_new):
                for f in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w_pos * sw
                    horiz_end = horiz_start + kw

                    # Slice
                    a_slice = A_prev_padded[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Gradients
                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, :] += dZ[i, h, w_pos, f] * W[:, :, :, f]
                    dW[:, :, :, f] += a_slice * dZ[i, h, w_pos, f]
                    db[0, 0, 0, f] += dZ[i, h, w_pos, f]

    # Remove padding if 'same'
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
