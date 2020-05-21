#!/usr/bin/env python3
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
################# TODO: separte variables in .env file?
USE_MQTT=False  # To be able to test without MQTT
DEFAULT_CONFIDENCE=0.5

DEFAULT_DEVICE="MYRIAD" #CPU
if DEFAULT_DEVICE == "CPU":
    DEFAULT_PREC = 32
else:
    DEFAULT_PREC=16

DEFAULT_MODEL=f"./models/intel/person-detection-retail-0013/FP{DEFAULT_PREC}/person-detection-retail-0013.xml"
#DEFAULT_MODEL="./tmp/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml" # Lot of errors
#DEFAULT_MODEL="./tmp/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.xml" # input_shape = [1, 3]
#DEFAULT_MODEL="./tmp/faster_rcnn_resnet101_ava_v2.1_2018_04_30/frozen_inference_graph.xml" # input_shape = [1, 3]
#DEFAULT_MODEL="./tmp/onnx-mobilenetv2-1.0/mobilenetv2-1.0.xml" # only for classification
#DEFAULT_MODEL="tmp/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.xml" # input_shape:  [1, 3]

#DEFAULT_MODEL="tmp/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml" ## Input (1,3)

#https://github.com/zlingkang/mobilenet_ssd_pedestrian_detection
#DEFAULT_MODEL="tmp/mobilenet_ssd_pedestrian_detection/MobileNetSSD_deploy10695.xml" # nothing detected

#https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd
#DEFAULT_MODEL="tmp/onnx/ssd-10.xml" #unsupported layers 'NonMaxSuppression' VPU

#https://github.com/weiliu89/caffe/tree/ssd
#https://drive.google.com/file/d/0BzKzrI_SkD1_WnR2T1BGVWlCZHM/view
#DEFAULT_MODEL="tmp/caffe/vggnet/VGG_VOC0712Plus_SSD_300x300_iter_240000.xml" # really slow; label = 15.0

DEFAULT_INPUT='resources/Pedestrian_Detect_2_1_1.mp4'


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str, default=DEFAULT_MODEL, 
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str, default=DEFAULT_INPUT,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default=DEFAULT_DEVICE,
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=DEFAULT_CONFIDENCE,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-o", "--output_video", type=str, default="out.mp4",
                        help="Name of the output video")
    parser.add_argument("-x", "--disable_video_output", type=bool, default=False,
                        help="Disable the output of key video frames to stdout\n"
                            "If enabled, you need to pipe the output of this script to ffmpeg\n"
                            f"e.g.: python3 {__file__} | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 1280x720 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm")
    parser.add_argument("-q", "--disable_mqtt", type=bool, default=False,
                        help="Disable the connection to MQTT server (for example, to test the inference only)")
    parser.add_argument("-s", "--show_window", type=bool, default=True,
                        help="Shows a Window with the processed output of the image or video")
    parser.add_argument("-k", "--skip_frames", type=int, default=0,
                        help="Skip # of frames on the start of the video.")
    return parser


def connect_mqtt():
    if USE_MQTT:
        client = mqtt.Client()
        client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
        return client

def disconnect_mqtt(client):
    if USE_MQTT:
        client.disconnect()

def draw_masks(result, frame, v_width, v_height, prob_threshold):
    '''
    Draw bounding boxes onto the frame.
    '''
    nr_people_on_frame = 0
    valid_boxes = []
    #print("Result: ", result)
    #print("shape: ", result.shape)

    def check_boundary(x, y, maxX=v_width):
        newX = x
        newY = y

        if y < 20:
            newY = 20
        if x > maxX - 50:
            newX = maxX - 50
        
        return newX, newY
    
    for box in result[0][0]: # Output shape is 1x1x200x7
        label = box[1]
        #if label != 1:  # Person Class in COCO
        #    continue

        confidence = box[2]
        if confidence >= prob_threshold:
            xmin = int(box[3] * v_width)
            ymin = int(box[4] * v_height)
            xmax = int(box[5] * v_width)
            ymax = int(box[6] * v_height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, f"{confidence:.4f}",check_boundary(xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), thickness=2)
            nr_people_on_frame += 1
            valid_boxes.append(box)
    return frame, nr_people_on_frame, valid_boxes

def draw_stats(frame, nr_people, total_people, duration, frame_nr):
    '''
    Draw statistics onto the frame.
    '''

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1
    x = 10
    y = 15

    def putText(text):
        nonlocal frame
        nonlocal y
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness=thickness)
        y += 20

    if frame_nr != 0:
        putText(f"Frame: {frame_nr}")

    putText(f"In Frame: {nr_people}")
    putText(f"Duration: {duration:.2f}s")
    putText(f"Total: {total_people}")
    
    return frame

def preprocess_frame(raw_frame, required_size):
    """
    Preprocess the frame according to the model needs.
    """
    frame = cv2.resize(raw_frame, required_size)
    #cv2.cvtColor if not BGR
    frame = frame.transpose((2,0,1))        #depends on the model
    frame = frame.reshape(1, *frame.shape)  #depends on the model
    return frame

def infer_on_stream(args, mqtt_client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    # --> it's in args.prob_threshold

    ### Load the model through `infer_network` ###
    network = infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    required_size = (net_input_shape[3], net_input_shape[2])

    ### Handle the input stream ###
    cap = cv2.VideoCapture(args.input) #, cv2.CAP_FFMPEG)
    cap.open(args.input)

    v_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #OpenCV 3+
    print(f"Total frames: {total_frames}. FPS: {fps}")

    if args.isImage:
        out = None
    else:
        print("Creating VideoWriter")
        fourCC = cv2.VideoWriter_fourcc(*'mp4v') #try with: 'MJPG', 'XVID', 'MP4V'
        out = cv2.VideoWriter(args.output_video, fourCC, fps, (v_width, v_height))

    ### Loop until stream is over ###
    frame_nr=0
    total_people_counted = 0
    previous_nr_people_on_frame = 0
    duration = 0
    duration_start = 0
    duration_end = 0

    while cap.isOpened:
        frame_nr += 1

        #print(f"Frame {frame_nr} of {total_frames}")

        ### Read from the video capture ###
        flag, raw_frame = cap.read()
        if not flag:
            break

        # skip frames (to help debug). Frame 62: the girl starts walking, frame 190: undetected and then redetected
        if args.skip_frames != 0 and frame_nr < args.skip_frames:
            continue

        ### Pre-process the image as needed ###
        frame = preprocess_frame(raw_frame, required_size)
        
        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(frame)

        ### Wait for the result ###
        if infer_network.wait() == 0:

            ### Get the results of the inference request ###
            result = infer_network.get_output() #[1,1,200,7]

            ### Extract any desired stats from the results ###
            out_frame, nr_people_on_frame, valid_boxes = draw_masks(result, raw_frame, v_width, v_height, args.prob_threshold)

            #if nr_people_on_frame is equal, update the duration (needs to be per-person)
            #else, there is someone new or someone less
            if previous_nr_people_on_frame == nr_people_on_frame:
                if duration_start != 0:
                    duration_end = time.perf_counter()
            else:
                if previous_nr_people_on_frame < nr_people_on_frame:
                    #new people on frame
                    #=== TODO: check if it was an error and reappears on next frame
                    total_people_counted += nr_people_on_frame - previous_nr_people_on_frame
                    duration_start = time.perf_counter()
                else:
                    #less people on frame
                    duration_end = time.perf_counter()
                    duration_start = 0

            previous_nr_people_on_frame = nr_people_on_frame

            person_stats = {'count': nr_people_on_frame, 'total': total_people_counted}

            duration = duration_end - duration_start if duration_start != 0 else 0
            print(f"frame: {frame_nr} ###### count: {nr_people_on_frame}, total: {total_people_counted}, duration: {duration}")
            
            out_frame = draw_stats(out_frame, nr_people_on_frame, total_people_counted, duration, frame_nr)

            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if USE_MQTT:
                mqtt_client.publish("person", json.dumps(person_stats))
                mqtt_client.publish("person/duration", json.dumps({'duration':duration}))

        ### Send the frame to the FFMPEG server ###
        if not args.disable_video_output:
            sys.stdout.buffer.write(out_frame)
            sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        if args.isImage:
            cv2.imwrite('output_image.jpg', out_frame)
        else:
            #print(f"Writing frame {actual_frame}")
            out.write(out_frame)

        if args.show_window:
            cv2.imshow('display', out_frame)

        key_pressed = cv2.waitKey(30)
        if key_pressed == 27 or key_pressed == 113: #Esc or q
            break #exit the while cap.isOpened() loop

        if key_pressed == 32: #if space: advance 20 frames
            args.skip_frames = frame_nr + 20

    if not args.isImage:
        out.release()

    cap.release()
    cv2.destroyAllWindows()

def sanitize_input(args):
    if args.input == "CAM" or args.input == "0":
        args.input = 0 #the webcam
    else:
        args.input = os.path.abspath(args.input)

    if args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        args.isImage = True
    else:
        args.isImage = False

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    sanitize_input(args)

    # Connect to the MQTT server
    USE_MQTT = not args.disable_mqtt
    mqtt_client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, mqtt_client)
    disconnect_mqtt(mqtt_client)


if __name__ == '__main__':
    main()
