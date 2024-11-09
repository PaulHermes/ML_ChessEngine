import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
import parameters
import util

class ReinforcementLearningModel:
    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.convolutional_filters = parameters.convolution_filters
        self.residual_blocks = parameters.residual_block_count
        self.kernel_size = (parameters.kernel_size, parameters.kernel_size)
        self.stride = (parameters.stride, parameters.stride)

        self.model = None

    # https://www.chessprogramming.org/AlphaZero#Network_Architecture
    # The policy head applies an additional rectified, batch-normalized convolutional layer,
    # followed by a final convolution of 73 filters for chess, with the final policy output represented as an 8x8 board array as well,
    # for every origin square up to 73 target square possibilities (NRayDirs x MaxRayLength + NKnightDirs + NPawnDirs * NMinorPromotions),
    # encoding a probability distribution over 64x73 = 4,672 possible moves, where illegal moves were masked out by setting their probabilities to zero,
    # re-normalising the probabilities for remaining moves. The value head applies an additional rectified,
    # batch-normalized convolution of 1 filter of kernel size 1x1 with stride 1, followed by a rectified linear layer of size 256 and a tanh-linear layer of size 1.
    def build(self, should_plot_model: bool = False, compile_model: bool = True):
        inputs = Input(shape=self.input_shape)

        # Initial convolutional layer (before residual blocks)
        # Standard practice is: Convolution → Batch Normalization → Activation
        x = layers.Conv2D(filters=self.convolutional_filters, kernel_size=self.kernel_size, strides=self.stride,
                          padding='same', activation=None)(inputs)
        x = layers.BatchNormalization()(x) # We normalize first
        x = layers.Activation('relu')(x) # Then we apply ReLU

        for _ in range(self.residual_blocks):
            x = self.residual_block(x)

            # Policy Head
            x_policy = layers.Conv2D(filters=self.convolutional_filters, kernel_size=self.kernel_size,
                                     strides=self.stride, padding='same')(x)
            x_policy = layers.BatchNormalization()(x_policy)
            x_policy = layers.Activation('relu')(x_policy)

            policy_head = layers.Conv2D(filters=parameters.possible_moves, kernel_size=(1, 1), strides=1, padding='same')(x_policy)
            policy_output = layers.Flatten()(policy_head)
            policy_output = layers.Dense(self.output_shape[0],activation='softmax', name='policy_head')(policy_output)

            # Value Head
            x_value = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)
            x_value = layers.BatchNormalization()(x_value)
            x_value = layers.Activation('relu')(x_value)

            # Flatten and apply dense layers
            x_value = layers.Flatten()(x_value)
            x_value = layers.Dense(256, activation='relu')(x_value)

            # Final dense layer with tanh activation for the value output
            value_output = layers.Dense(1, activation='tanh', name='value_head')(x_value)

            # Create the model
            self.model = models.Model(inputs=inputs, outputs=[policy_output, value_output])

            if compile_model:
                self.compile_model()

            if should_plot_model:
                tf.keras.utils.plot_model(self.model, to_file='images/model.png', show_shapes=True, show_layer_names=True)

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=util.get_learning_rate_schedule("warmup"), decay=parameters.weight_decay,
                           beta_1=parameters.adam_beta_1, beta_2=parameters.adam_beta_2),
            loss={'policy_head': 'categorical_crossentropy', 'value_head': 'mean_squared_error'},
            loss_weights={'policy_head': 1.0, 'value_head': 1.0}
        )


    def residual_block(self, x):
        shortcut = x  # Save the input for the skip connection

        # First Convolutional layer
        x = layers.Conv2D(filters=self.convolutional_filters, kernel_size=self.kernel_size, strides=self.stride,
                          padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Second Convolutional layer
        x = layers.Conv2D(filters=self.convolutional_filters, kernel_size=self.kernel_size, strides=self.stride,
                          padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)

        # Add the skip connection
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)

        return x