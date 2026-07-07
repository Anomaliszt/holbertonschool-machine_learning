#!/usr/bin/env python3
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using Keras.
    
    Args:
        X: K.Input of shape (m, 28, 28, 1)
    
    Returns:
        A compiled K.Model
    """
    
    initializer = K.initializers.HeNormal(seed=0)
    
    # Input
    inputs = X
    
    # Layer 1: Conv2D (6 filters, 5x5, same padding)
    x = K.layers.Conv2D(6, (5, 5), padding='same', 
                        kernel_initializer=initializer, 
                        activation='relu')(inputs)
    
    # Layer 2: MaxPooling 2x2 with stride 2
    x = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Layer 3: Conv2D (16 filters, 5x5, valid padding)
    x = K.layers.Conv2D(16, (5, 5), padding='valid', 
                        kernel_initializer=initializer, 
                        activation='relu')(x)
    
    # Layer 4: MaxPooling 2x2 with stride 2
    x = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Flatten
    x = K.layers.Flatten()(x)
    
    # Layer 5: Fully Connected (120 nodes)
    x = K.layers.Dense(120, kernel_initializer=initializer, 
                       activation='relu')(x)
    
    # Layer 6: Fully Connected (84 nodes)
    x = K.layers.Dense(84, kernel_initializer=initializer, 
                       activation='relu')(x)
    
    # Output Layer: Softmax (10 nodes)
    outputs = K.layers.Dense(10, kernel_initializer=initializer, 
                             activation='softmax')(x)
    
    # Build the model
    model = K.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
