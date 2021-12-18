import numpy as np
import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add
from tensorflow.keras.models import Model

rgb_mean = np.array([0.4488, 0.4371, 0.4040]) * 255


class WDSR:
    def __init__(self, res_blocks, scale_factor):
        self.model_input = Input(shape=(None, None, 3))
        self.model = Conv2D(32, kernel_size=(3, 3), padding="same")(self.model_input)
        for _ in range(res_blocks):
            current_model_state = self.model
            self.model = Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu")(self.model)
            self.model = Conv2D(32, kernel_size=(3, 3), padding="same")(self.model)
            self.model = Add()([current_model_state, self.model])
        self.model = Conv2D(3 * (scale_factor ** 2), kernel_size=(3, 3), padding="same")(self.model)
        self.model_skip_connection = Conv2D(3 * (scale_factor ** 2), kernel_size=(5, 5), padding="same")(
            self.model_input
        )
        self.model = Add()([self.model, self.model_skip_connection])
        self.model = Conv2D(3, kernel_size=(3, 3), padding="same")(self.model)
        self.model = Model(self.model_input, self.model)

    def normalize(feature_map):
        rgb_mean = np.array([0.4488, 0.4371, 0.4040]) * 255
        return (feature_map - rgb_mean) / 127.5

    def denormalize(feature_map):
        rgb_mean = np.array([0.4488, 0.4371, 0.4040]) * 255
        return feature_map * 127.5 + rgb_mean

    def pixel_shuffle(scale):
        return lambda x: tf.nn.depth_to_space(x, scale)

    def get_model(self):
        return self.model
