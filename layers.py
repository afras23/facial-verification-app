# layers.py
# Custom L1 distance layer module required for loading the custom model

# Install dependencies 
import tensorflow as tf
from keras.layers import Layer

class L1Dist(Layer): 

    #Custom L1 (Manhattan) distance layer for neural networks.
    #Computes the element-wise absolute difference between two embeddings.
    #Required for loading models that include this custom layer.

    def __init__(self, **kwargs): 

        # Initializes the custom L1Dist layer.

        super().__init__()
        # Args: **kwargs: Additional keyword arguments for layer configuration.

    def call(self, input_embedding, validation_embedding): 
        
        # Defines the computation for the layer.

        # Args:
            # input_embedding (Tensor): Embedding from input image.
            # validation_embedding (Tensor): Embedding from validation/reference image.

        # Returns:
            # Tensor: Element-wise absolute difference between input and validation embeddings.
            
        return tf.math.abs(input_embedding - validation_embedding) 