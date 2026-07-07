#!/usr/bin/env python3
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Output dimensions
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
        h_out = ((h_prev - kh) // sh) + 1
        w_out = ((w_prev - kw) // sw) + 1

        pad_top = pad_bottom = 0
        pad_left = pad_right = 0

    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Padding
    A_pad = np.pad(
        A_prev,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
    )

    Z = np.zeros((m, h_out, w_out, c_new))

    # Convolution
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh

            w_start = j * sw
            w_end = w_start + kw

            window = A_pad[:, h_start:h_end, w_start:w_end, :]

            for c in range(c_new):
                Z[:, i, j, c] = (
                    np.sum(window * W[:, :, :, c], axis=(1, 2, 3)) + b[0, 0, 0, c]
                )

    return activation(Z)
