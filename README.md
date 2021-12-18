# Comparison of Image Super Resolution Models on Edge Devices

This is the final project GitHub repository for COMS 6998: Practical Deep Learning Systems Performance, by team members Abinai Pasunuri (ap4136) and Suraj Geddam (sg3885).

## Introduction

Super resolution is an interesting problem that has been around for quite some time. Maybe the most notable example being surveillance photos of criminals being "zoom-and-enhance"d in CSI and similar TV shows. However, with deep neural nets, it seems like this sci-fi-ish technology is more and more becoming a reality, hence we were interested in looking into different models used to actually perform super resolution. Also, typically when we think deep learning we consider massive clusters with large amounts of GPU compute power. But we were interested in how an ordinary person might get to use this technology. So we measured the performance of running super resolution (from an already trained model) converted to run client-side on a web application or on a phone directly without having to leverage cloud computing resources.

## Technical details

We used three SR architectures - SRCNN, EDSR and WDSR. The networks were trained using TensorFlow and Keras on Python, on a Google Cloud instance with a single P100 GPU. The dataset used was custom-built and it is a combination of the General100 Super Resolution dataset and a subset of COCO, with 1500 images (960 train, 240 validation, 300 test). Then, they were converted to TFLite and TF.js versions so that they could run inferencing directly from an Android device or a browser respectively. For the same purpose, we have built an Android app using Android Studio and a webapp in plain HTML and JavaScript (with TensorFlow.js included).

## Code

We have included the Python code used for training the models, as well as the source code of the webapp and the Android app we used to test performance.

## Commands

You can run training by

    python main.py --model "<MODEL_NAME>"

This gives .h5 format trained models. <MODEL_NAME> can either be "srcnn", "wdsr", or "edsr"
To convert them to TF.js format for the webapp (note: you need TF.js installed), use

    tensorflowjs_converter --input_format keras <model name>.h5 <output directory>

To convert them to TFLite format for the Android app, use

    tflite_convert --keras_model_file=/path/to/model.h5 --output_file=/path/to/output.tflite

For the webapp, you need to serve the model files from a local HTTP server. We used livehttp for this purpose, but any simple HTTP fileserver should work as long as it supports CORS.
For the Android app, first clone the repository:

    git clone whatever

Then you can import the project into Android Studio directly and build/run on your Android device of choice.

## Results

### Training Results

| Model Type | Mean Squared Error |  PNSR   | Inference Time (milliseconds) | Throughput (Img/sec) | Model Size (MB) |
| :--------: | :----------------: | :-----: | :---------------------------: | :------------------: | :-------------: |
|   SRCNN    |      59.3088       | 30.4982 |            6.6667             |         150          |      1.322      |
|    EDSR    |      63.5052       | 30.5413 |            88.6666            |       11.2782        |      7.610      |
|    WDSR    |      59.6232       | 30.5200 |            66.6667            |          15          |      5.517      |

### Inference Results on Web Application

| Quantization |         ->          | 8-bit Integer Quantization |       <-        |         ->          | 16-bit Float Quantization |       <-        |
| :----------: | :-----------------: | :------------------------: | :-------------: | :-----------------: | :-----------------------: | :-------------: |
|  Model Type  | Inference Time (ms) |    Throughput (Img/sec)    | Model Size (KB) | Inference Time (ms) |   Throughput (Img/sec)    | Model Size (KB) |
|    SRCNN     |        2.64         |          378.7879          |       108       |        2.92         |         342.4665          |       215       |
|     EDSR     |         9.7         |          103.0928          |       622       |        12.6         |          82.2368          |      1244       |
|     WDSR     |        9.54         |          104.822           |       439       |       10.1999       |          98.0392          |       877       |

### Inference Results on Android Device

| Quantization |         ->          | 8-bit Integer Quantization |       <-        |         ->          | 16-bit Float Quantization |       <-        |
| :----------: | :-----------------: | :------------------------: | :-------------: | :-----------------: | :-----------------------: | :-------------: |
|  Model Type  | Inference Time (ms) |    Throughput (Img/sec)    | Model Size (KB) | Inference Time (ms) |   Throughput (Img/sec)    | Model Size (KB) |
|    SRCNN     |          1          |            1000            |       114       |       1.4972        |         667.9134          |       218       |
|     EDSR     |       1.7186        |          581.8690          |       658       |       6.4171        |         155.8336          |      1257       |
|     WDSR     |       1.7603        |          568.085           |       476       |       3.1287        |         319.6216          |       890       |

## References

Part of the code used in this project is influenced and taken from the following GitHub Repos that have done similar implementations of Super-Resolution models.

- https://github.com/MarkPrecursor/SRCNN-keras
- https://github.com/krasserm/super-resolution
