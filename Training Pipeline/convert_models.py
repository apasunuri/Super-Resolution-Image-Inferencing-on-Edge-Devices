import tensorflow as tf

from main import psnr

model_dir = ""
model = tf.keras.models.load_model(model_dir, custom_objects={"psnr": psnr})
output_dir = ""
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open(output_dir, "wb") as f:
    f.write(tflite_model)
