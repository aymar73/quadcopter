#https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/
#https://keras.io/layers/merge/
from keras import layers, models, optimizers, initializers
from keras import backend as K

class KerasNet:

    def __init__(self, state_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
        """
        self.state_size = state_size
        
        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        # Define input layer
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layer
        net_states = layers.Dense(units=1, kernel_initializer=initializers.Ones(), 
            bias_initializer=initializers.Zeros(), activation=None)(states)

        # Add final output
        #net_states = layers.BatchNormalization()(net_states)
        value = layers.Activation('relu')(net_states)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=value)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')


