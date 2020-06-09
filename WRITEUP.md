# Project Write-Up
## People Counter App at the Edge

This is the "People Counter App at the Edge". It uses the Intel® OpenVINO™ Toolkit to detect and count the people that appears in the video and shows statistics on a website.

## Custom Layers

When talking about layers, we are talking about math functions that are needed to run Deep Learning (Artificial Intelligence) Models. More specifically: "The abstract concept of a math function that is selected for a specific purpose (relu, sigmoid, tanh, convolutional). This is one of a sequential series of building blocks within the neural network"[1]

The OpenVINO™ Toolkit supports a great number of [Layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) from the different supported neural network frameworks.

However, the Deep Learning frameworks are not static projects. They evolve over time and add new layers as needed.
So, we need a process to convert the layers that are unsupported by OpenVINO™.

OpenVINO™ supports these "unknown" layers indirectly, by so-called "Custom Layers". There is an official Guide to [convert custom layers](https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html) where it explains in detail how to do it. There is also a (tutorial)[https://github.com/david-drew/OpenVINO-Custom-Layers/tree/master/2019.r2.0] on how to convert the layers in Linux and Windows.

But, as a summary, we need to add extensions to both the Model Optimizer and the Inference Engine.

First, we need to generate the Extension Template Files using the Model Extension Generator ($INTEL_OPENVINO_DIR/deployment_tools/tools/extension_generator/extgen.py), then we need to edit the code stubs with our code (some in Python and others in C++), compile C++ code and generate the IR files. Then, it's ready to be used by the Inference Engine. Please see the tutorial for a more step-by-step explanation.

## Comparing Model Performance

My method to compare models before and after conversion to Intermediate Representations were: to run the original model in the original framework, then converting the model to OpenVINO IR and running on OpenVINO.

The difference between model accuracy pre- and post-conversion was: 

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

[1]: https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html