# Project Write-Up
## People Counter App at the Edge

This is the "People Counter App at the Edge". It uses the Intel® OpenVINO™ Toolkit to detect and count the people that appears in the video and shows statistics on a website.

Many AI models were tested to make this software, but finally the "original" *person-detection-retail-0013* was selected, as it is more accurate and performant.

## Custom Layers

When talking about layers, we are talking about math functions that are needed to run Deep Learning (Artificial Intelligence) Models. More specifically: "The abstract concept of a math function that is selected for a specific purpose (relu, sigmoid, tanh, convolutional). This is one of a sequential series of building blocks within the neural network"[1]

The OpenVINO™ Toolkit supports a great number of [Layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) from the different supported neural network frameworks.

However, the Deep Learning frameworks are not static projects. They evolve over time and add new layers as needed.
So, we need a process to convert the layers that are unsupported by OpenVINO™.

OpenVINO™ supports these "unknown" layers indirectly, by so-called "Custom Layers". There is an official Guide to [convert custom layers](https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html) where it explains in detail how to do it. There is also a [tutorial](https://github.com/david-drew/OpenVINO-Custom-Layers/tree/master/2019.r2.0) on how to convert the layers in Linux and Windows.

But, as a summary, we need to add extensions to both the Model Optimizer and the Inference Engine.

First, we need to generate the Extension Template Files using the Model Extension Generator ($INTEL_OPENVINO_DIR/deployment_tools/tools/extension_generator/extgen.py), then we need to edit the code stubs with our code (some in Python and others in C++), compile C++ code and generate the IR files. Then, it's ready to be used by the Inference Engine. Please see the tutorial for a more step-by-step explanation.

## Comparing Model Performance

I was not able to directly compare the models with and without OpenVINO in the same conditions, as I don't have a supported machine. My laptop is a 3rd Gen Intel i7, and I have a Neural Compute Stick 2 (NCS2). As the NCS2 is able to run in a Raspberry Pi, I found a way to run it in my 3rd Gen i7.

########################################################

My method to compare models before and after conversion to Intermediate Representations was to run the original model in the original framework, measure the performance and accuracy, then converting the model to OpenVINO IR and running on OpenVINO, and get the new numbers to compare.

The difference between model accuracy pre- and post-conversion was:
* TensorFlow pretrained 

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
* Controlling qeues (for example, in supermarkets, in banks, etc) and emit an alert to help to "balance" them. This would be useful to make the overall system more efficient and to serve the customers faster.

* Count if there is more people than permited in the same space, avoiding a crowd of people. This would be useful to avoid the spread of the Coronavirus Disease (Covid19).

* In a bus station, a people counter app could help to get more busses on road according to the quantity of people waiting for the next bus. This would be useful to get a faster service for the passengers.

## Assess Effects on End User Needs

I have tested the different models in different scenarios of lighting, cameras (webcam, surveillance camera, professionally filmed youtube videos, etc) and image sizes, resulting in differences on model accuracy.

The accuracy of the models are affected the most by poor lighting, making it necessary to adjust the algorithm to correctly counting people. Also, the quality of the camera makes a huge difference in the accuracy of the models.

I think that the best way to assure good model accuracy in poor lighting or camera quality conditions is to train a specific model for these scenarios.

## Model Research

In investigating potential people counter models, I tried each of the following models (as listed in models.py, mainly downloaded by scripts/model_conversion.sh):

- Model 1: [ssd_mobilenet_v2_coco]
  - Model Source: [TensorFlow ModelZoo](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
    ```
    "$MO --input_model frozen_inference_graph.pb --data_type=FP16 --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config $MO_DIR/extensions/front/tf/ssd_v2_support.json"
    ```
    Where $MO is pointing to the path of the model optimizer (mo.py) and $MO_DIR is the root directory of the model optimizer.
    I had to make it FP16, as I am using an NCS2. However, I have noticed that it also works if I run *converted models* as FP32 on NCS2. For pretrained Intel models, FP32 models do not work.

    To see how I have downloaded the model, uncompressed it, changed to the correct folders, etc., please see `scripts/model_conversion.sh`

  - The model was insufficient for the app because: there where a lot of undetected frames (frames not detecting people even when there were people in the frame, even when lowering the detection threshold), making it hard to maintain a good counter. Also, the 2nd person in the test video is really hard to detect in this model (and in the majority of the tested models).

  - I tried to improve the model for the app by lowering the detection threshold, filtering by person class, calculate the times when it does not detect a person and try to make an algorithm to reduce the jitter.
  
- Model 2: [faster_rcnn_resnet50_coco_2018_01_28]
  - Model Source: [TensorFlow ModelZoo](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
     ```
    "$MO --input_model frozen_inference_graph.pb --keep_shape_ops --data_type=FP16 --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config $MO_DIR/extensions/front/tf/faster_rcnn_support.json"
    ```

  - The model was insufficient for the app because it was somewhat slow compared to the "native" **person-detection-retail-0013**. Also, it detected "more people" as the person was going out to the right.

  - I tried to improve the model for the app by lowering the detection threshold, filtering by person class, calculate the times when it does not detect a person and try to make an algorithm to reduce the jitter.

- Model 3: [caffe-vggnet-ssd300]
  - Model Source: [Zip file obtained from a blog post](https://drive.google.com/file/d/0BzKzrI_SkD1_WnR2T1BGVWlCZHM/view)
  - I converted the model to an Intermediate Representation with the following arguments:
    `$MO --input_model VGG_VOC0712Plus_SSD_300x300_iter_240000.caffemodel --input_proto deploy.prototxt`
  - The model was insufficient for the app because it was really slow and had a lot of glitches
  - I tried to improve the model for the app by lowering the detection threshold, filtering by person class, calculate the times when it does not detect a person and try to make an algorithm to reduce the jitter.

Note: there were more tested models. Some of them are listed in models.py. The models that I really wanted to test were of GluonCV (MXNet), but they are not supported. It was really hard to find a model in MXNet compatible with OpenVINO.

[1]: https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html