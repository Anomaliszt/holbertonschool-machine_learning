#!/usr/bin/env python3
import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network.
    """
    # Retrieve dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape  # _ should be c_prev
    
    sh, sw = stride
    
    # Compute output dimensions and padding
    if padding.lower() == "same":
        h_new = int(np.ceil(float(h_prev) / sh))
        w_new = int(np.ceil(float(w_prev) / sw))
        pad_h_total = max((h_new - 1) * sh + kh - h_prev, 0)
        pad_w_total = max((w_new - 1) * sw + kw - w_prev, 0)
        pad_h = pad_h_total // 2
        pad_w = pad_w_total // 2
    elif padding.lower() == "valid":
        h_new = int(np.floor((h_prev - kh) / sh)) + 1
        w_new = int(np.floor((w_prev - kw) / sw)) + 1
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")
    
    # Pad A_prev
    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    
    # Initialize Z
    Z = np.zeros((m, h_new, w_new, c_new))
    
    # Perform convolution
    for i in range(m):  # loop over batch
        for h in range(h_new):
            for ww in range(w_new):
                for c in range(c_new):
                    # Define slice boundaries
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = ww * sw
                    horiz_end = horiz_start + kw
                    
                    # Extract slice
                    a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Element-wise multiply and sum + bias
                    Z[i, h, ww, c] = np.sum(a_slice_prev * W[:, :, :, c]) + b[0, 0, 0, c]
    
    # Apply activation
    A = activation(Z)
    
    return A
