#!/usr/bin/env python3
import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a neural network."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        # Calculate output dimensions
        h_new = int(np.ceil(h_prev / sh))
        w_new = int(np.ceil(w_prev / sw))
        # Calculate padding
        ph_total = (h_new - 1) * sh + kh - h_prev
        pw_total = (w_new - 1) * sw + kw - w_prev
        ph = ph_total // 2
        pw = pw_total // 2
        # Pad more on the end if odd
        pad_top = ph
        pad_bottom = ph_total - ph
        pad_left = pw
        pad_right = pw_total - pw
        A_prev_padded = np.pad(A_prev, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    elif padding == "valid":
        ph = pw = 0
        h_new = (h_prev - kh) // sh + 1
        w_new = (w_prev - kw) // sw + 1
        A_prev_padded = A_prev
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    a_slice = A_prev_padded[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[0, 0, 0, c]

    A = activation(Z)
    return A
