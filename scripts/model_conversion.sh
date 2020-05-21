#!/bin/bash
#
# Script to convert a TensorFlow model to Intel OpenVINO IR
#

# Download a TensorFlow model
TF_MODEL_NAME=ssd_mobilenet_v2_coco_2018_03_29
#TF_MODEL_NAME=faster_rcnn_resnet50_coco_2018_01_28
#TF_MODEL_NAME=faster_rcnn_resnet101_kitti_2018_01_28
#TF_MODEL_NAME=faster_rcnn_resnet101_ava_v2.1_2018_04_30
#TF_MODEL_NAME=faster_rcnn_inception_v2_coco_2018_01_28
TF_MODEL_FILE=${TF_MODEL_NAME}.tar.gz
TF_MODEL=http://download.tensorflow.org/models/object_detection/$TF_MODEL_FILE

MODEL_TYPE=ssd_v2
#MODEL_TYPE=faster_rcnn

DOWNLOAD_DIR=`realpath .`/../tmp

mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

if [ ! -f $DOWNLOAD_DIR/$TF_MODEL_FILE ]; then
	wget $TF_MODEL
	if [ $? -ne 0 ]; then
		echo "No se pudo descargar el archivo"
		exit 1
	fi
else
	echo "File $TF_MODEL_FILE already exists"
fi

MODEL_DIR=$DOWNLOAD_DIR/$TF_MODEL_NAME

if [ ! -f $MODEL_DIR/frozen_inference_graph.pb ]; then
	tar xvzf $TF_MODEL_FILE
fi


# Install prerequisites for Model Optimizer:
if [ -z "$INTEL_OPENVINO_DIR" ]; then
	source /opt/intel/openvino/bin/setupvars.sh
fi

MO_DIR=/opt/intel/openvino/deployment_tools/model_optimizer
MO=$MO_DIR/mo.py

#cd $MO_DIR/install_prerequisites/
#./install_prerequisites_tf.sh
# or better: pip install -r $MO_DIR/requirements_tf.txt

cd $MODEL_DIR

# Model Optimizer
$MO --input_model frozen_inference_graph.pb \
	--tensorflow_object_detection_api_pipeline_config pipeline.config \
	--reverse_input_channels --tensorflow_use_custom_operations_config \
	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/${MODEL_TYPE}_support.json


##### end.

#ONNX: https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz
# conv: $MO_DIR/mo.py --input_model mobilenetv2-1.0.onnx (https://github.com/onnx/models/tree/master/vision/classification/mobilenet)

#/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name mobilenet-ssd -o raw_models

#### Model Zoos:
# TensorFlow: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# Caffe: https://github.com/BVLC/caffe/wiki/Model-Zoo
# ONNX: https://github.com/onnx/models
# MXNet: https://gluon-cv.mxnet.io/model_zoo/index.html
# https://modelzoo.co/

#### OpenVINO pretrained models:
# https://docs.openvinotoolkit.org/2019_R1/_docs_Pre_Trained_Models.html
