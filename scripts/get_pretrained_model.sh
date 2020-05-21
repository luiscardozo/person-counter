#!/bin/bash
#

# get the pretrained model, to test the code
#

DOWNLOADER_DIR=$INTEL_OPENVINO_DIR/deployment_tools/open_model_zoo/tools/downloader
DOWNLOADER=$DOWNLOADER_DIR/downloader.py
MODEL=person-detection-retail-0013
PRECISIONS=FP32,FP16
MODELS_DIR=../models
PYTHON=python3
REQS=$DOWNLOADER_DIR/requirements.in

if [ -z "$INTEL_OPENVINO_DIR" ]; then
    source /opt/intel/openvino/bin/setupvars.sh
fi

$PYTHON -c "import requests"
if [ $? -eq 1 ]; then
    echo "Downloader dependencies not installed. Install them? (y/whatever)"
    read resp
    if [ "$resp" == 'y' ]; then
        $PYTHON -m pip install -r $REQS
    fi
fi

if [ ! -d $MODELS_DIR ]; then
    mkdir -p $MODELS_DIR
fi

$DOWNLOADER -o $MODELS_DIR --name $MODEL --precisions $PRECISIONS
