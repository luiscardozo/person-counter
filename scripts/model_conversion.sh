#!/bin/bash
#
# Script to convert a TensorFlow model to Intel OpenVINO IR
#

# Download a TensorFlow model
#TF_MODEL_NAME=ssd_mobilenet_v2_coco_2018_03_29
#TF_MODEL_NAME=ssd_inception_v2_coco_2018_01_28
TF_MODEL_NAME=faster_rcnn_resnet101_coco_2018_01_28
#TF_MODEL_NAME=faster_rcnn_resnet50_coco_2018_01_28
#TF_MODEL_NAME=faster_rcnn_resnet101_kitti_2018_01_28
#TF_MODEL_NAME=faster_rcnn_resnet101_ava_v2.1_2018_04_30
#TF_MODEL_NAME=faster_rcnn_inception_v2_coco_2018_01_28
TF_MODEL_FILE=${TF_MODEL_NAME}.tar.gz
TF_MODEL=http://download.tensorflow.org/models/object_detection/$TF_MODEL_FILE

#MODEL_TYPE=ssd_v2
MODEL_TYPE=faster_rcnn

OPTS='--reverse_input_channels --data_type=FP16'

if [ "MODEL_TYPE" == 'faster_rcnn' ]; then
	#OPTS="$OPTS --output=detection_boxes,detection_scores,num_detections"
	OPTS="$OPTS --keep_shape_ops"
fi

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
CMD="$MO --input_model frozen_inference_graph.pb \
	$OPTS --tensorflow_object_detection_api_pipeline_config pipeline.config \
	--reverse_input_channels --tensorflow_use_custom_operations_config \
	$MO_DIR/extensions/front/tf/${MODEL_TYPE}_support.json"

echo "Ejecutar: $CMD"
$CMD
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

# faster_rcnn_resnet101_coco according to
# https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit/topic/816416
# MO=/opt/intel/openvino/deployment_tools/model_optimizer/mo.py
# $MO --framework tf --input_model frozen_inference_graph.pb \
#  --output=detection_boxes,detection_scores,num_detections \
#  --tensorflow_use_custom_operations_config  $MO_DIR/extensions/front/tf/${MODEL_TYPE}_support.json
#  --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels

# bug 'image_info' vs 'image_tensor' confirmed https://github.com/openvinotoolkit/openvino/issues/128
#https://github.com/opencv/cvat/pull/541
#https://github.com/opencv/cvat/pull/545
#
# tried with object_detection_demo_ssd_async
# https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit/topic/844282
# in /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_ssd_async

# Labels COCO: https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt

# Trying to convert according to:
# https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html#tf_od_custom_input_shape
# and
# https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_TensorFlow_Faster_RCNN_ObjectDetection_API.html

# mxnet: $MO --input_model faster_rcnn_resnet50_v1b_voc-0000.params
