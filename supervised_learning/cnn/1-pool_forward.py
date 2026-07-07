#!/usr/bin/env python3
import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape  # c_prev_w should match c_prev
    sh, sw = stride
    
    # Compute padding
    if padding == "same":
        # Calculate padding for same
        pad_h = ((h_prev - 1) * sh + kh - h_prev + 1) // 2 if sh == 1 else 0  # Simplified, but better to compute properly
        pad_w = ((w_prev - 1) * sw + kw - w_prev + 1) // 2 if sw == 1 else 0
        # Standard way for same padding (assuming stride consideration)
        # For general case:
        h_new = (h_prev + 2 * pad_h - kh) // sh + 1  # we'll set pad later
    elif padding == "valid":
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")
    
    # Better computation for padding in 'same'
    if padding == "same":
        # Pad so that output height/width matches ceil(input / stride)
        h_new = int(np.ceil(h_prev / sh))
        w_new = int(np.ceil(w_prev / sw))
        pad_h_total = (h_new - 1) * sh + kh - h_prev
        pad_w_total = (w_new - 1) * sw + kw - w_prev
        pad_h = pad_h_total // 2
        pad_w = pad_w_total // 2
    else:
        h_new = int(np.floor((h_prev - kh) / sh)) + 1
        w_new = int(np.floor((w_prev - kw) / sw)) + 1
        pad_h = 0
        pad_w = 0
    
    # Pad the input
    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    
    # Initialize output
    Z = np.zeros((m, h_new, w_new, c_new))
    
    # Convolution
    for i in range(m):  # loop over batch
        for h in range(h_new):  # loop over output height
            for w in range(w_new):  # loop over output width
                for c in range(c_new):  # loop over output channels
                    # Find the slice
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    
                    # Extract slice
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Convolve: sum over element-wise product with kernel
                    Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[0, 0, 0, c]
    
    # Apply activation
    A = activation(Z)
    
    return A
