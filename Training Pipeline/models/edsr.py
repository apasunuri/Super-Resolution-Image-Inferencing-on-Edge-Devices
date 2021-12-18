import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Lambda
from tensorflow.keras.models import Model


class EDSR:
    def __init__(self, res_blocks, scale_factor):
        self.model_input = Input(shape=(None, None, 3))
        self.model = Conv2D(64, kernel_size=(3, 3), padding="same")(self.model_input)
        for _ in range(res_blocks):
            current_model_state = self.model
            self.model_1 = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(current_model_state)
            self.model_1 = Conv2D(64, kernel_size=(3, 3), padding="same")(self.model_1)
            self.model = Add()([current_model_state, self.model_1])
        self.model_conv = Conv2D(64, kernel_size=(3, 3), padding="same")(self.model)
        self.model = Add()([self.model, self.model_conv])
        self.model = Conv2D(64 * (scale_factor ** 2), kernel_size=(3, 3), padding="same")(self.model)
        self.model = Conv2D(3, kernel_size=(3, 3), padding="same")(self.model)
        self.model = Model(self.model_input, self.model)

    def get_model(self):
        return self.model
