from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.models import Model


class SRCNN:
    def __init__(self):
        self.model_input = Input(shape=(None, None, 3))
        self.model = Conv2D(
            128, kernel_size=(9, 9), activation="relu", padding="same", kernel_initializer="glorot_uniform"
        )(self.model_input)
        self.model = Conv2D(
            64, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform"
        )(self.model)
        self.model = Conv2D(
            3, kernel_size=(5, 5), activation="linear", padding="same", kernel_initializer="glorot_uniform"
        )(self.model)
        self.model = Model(self.model_input, self.model)

    def get_model(self):
        return self.model
