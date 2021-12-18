import pickle

import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from data import DataLoader
from models.srcnn import SRCNN
from models.edsr import EDSR
from models.wdsr import WDSR

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


def psnr(Y_true, Y_pred):
    diff = tf.keras.layers.subtract([Y_true, Y_pred])
    diff = K.flatten(diff)
    rmse = K.sqrt(K.mean(K.pow(diff, 2)))
    rmse = 255.0 / rmse
    rmse_log = K.log(rmse) / K.log(tf.constant(10, rmse.dtype))
    return 20 * rmse_log


def train_srcnn(ids, learning_rate, epochs, batch_size):
    model_type = "srcnn"
    train_dataloader = DataLoader(ids[0], model_type, batch_size=batch_size)
    validation_dataloader = DataLoader(ids[1], model_type, batch_size=batch_size)
    test_dataloader = DataLoader(ids[2], model_type, batch_size=batch_size)
    checkpoint = ModelCheckpoint(
        filepath="./Saved Models/TensorFlow/Checkpoints/SRCNN/srcnn_{epoch:02d}.h5", monitor="val_loss"
    )
    model = SRCNN().get_model()
    model.compile(Adam(lr=learning_rate), loss="mean_squared_error", metrics=["mean_squared_error", psnr])
    model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        validation_data=validation_dataloader,
        validation_steps=len(validation_dataloader),
        callbacks=[checkpoint],
    )
    model.evaluate(test_dataloader, steps=len(test_dataloader))
    model.save("./Saved Models/TensorFlow/SRCNN/srcnn_final.h5")


def train_edsr(ids, learning_rate, epochs, batch_size, res_blocks=6, scale_factor=2):
    model_type = "edsr"
    train_dataloader = DataLoader(ids[0], model_type, batch_size=batch_size)
    validation_dataloader = DataLoader(ids[1], model_type, batch_size=batch_size)
    test_dataloader = DataLoader(ids[2], model_type, batch_size=batch_size)
    checkpoint = ModelCheckpoint(
        filepath="./Saved Models/TensorFlow/Checkpoints/EDSR/edsr_{epoch:02d}.h5", monitor="val_loss"
    )
    model = EDSR(res_blocks=res_blocks, scale_factor=scale_factor).get_model()
    model.compile(Adam(lr=learning_rate), loss="mean_squared_error", metrics=["mean_squared_error", psnr])
    model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        validation_data=validation_dataloader,
        validation_steps=len(validation_dataloader),
        callbacks=[checkpoint],
    )
    model.evaluate(test_dataloader, steps=len(test_dataloader))
    model.save("./Saved Models/TensorFlow/EDSR/edsr_final.h5")


def train_wdsr(ids, learning_rate, epochs, batch_size, res_blocks=6, scale_factor=2):
    model_type = "wdsr"
    train_dataloader = DataLoader(ids[0], model_type, batch_size=batch_size)
    validation_dataloader = DataLoader(ids[1], model_type, batch_size=batch_size)
    test_dataloader = DataLoader(ids[2], model_type, batch_size=batch_size)
    checkpoint = ModelCheckpoint(
        filepath="./Saved Models/TensorFlow/Checkpoints/WDSR/wdsr_{epoch:02d}.h5", monitor="val_loss"
    )
    model = WDSR(res_blocks=res_blocks, scale_factor=scale_factor).get_model()
    model.compile(Adam(lr=learning_rate), loss="mean_squared_error", metrics=["mean_squared_error", psnr])
    model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        validation_data=validation_dataloader,
        validation_steps=len(validation_dataloader),
        callbacks=[checkpoint],
    )
    model.evaluate(test_dataloader, steps=len(test_dataloader))
    model.save("./Saved Models/TensorFlow/WDSR/wdsr_final_1.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print(tf.test.is_gpu_available())
    parser.add_argument("--model", type=str, required=True, choices={"srcnn", "edsr", "wdsr"})
    args = parser.parse_args()
    model_type = args.model
    with open("./Data/data_splits.pickle", "rb") as file:
        data_split = pickle.load(file)
    train_ids = data_split["train"]
    validation_ids = data_split["validation_ids"]
    test_ids = data_split["test_ids"]
    if model_type == "srcnn":
        train_srcnn([train_ids, validation_ids, test_ids], 0.0003, 50, 32)
    elif model_type == "edsr":
        train_edsr([train_ids, validation_ids, test_ids], 0.001, 25, 2)
    elif model_type == "wdsr":
        train_wdsr([train_ids, validation_ids, test_ids], 0.001, 25, 8)
