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
from datetime import datetime as dt

import logging as log
import paho.mqtt.client as mqtt
import pafy

from argparse import ArgumentParser
from inference import Network
from models import Model

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
################# TODO: separte variables in .env file?
USE_MQTT=False  # To be able to test without MQTT
DEFAULT_CONFIDENCE=0.5
DEFAULT_LOGFILE="logs/infer_and_publish.log"
DEFAULT_LOGLEVEL="DEBUG"

DEFAULT_DEVICE="MYRIAD" #CPU
if DEFAULT_DEVICE == "CPU":
    DEFAULT_PREC = 32
else:
    DEFAULT_PREC=16

DEFAULT_INPUT='resources/Pedestrian_Detect_2_1_1.mp4'
isImage = False
allow_print = False #if --dev: print() else log.debug()

model = Model().get_default()
ini_time = time.perf_counter()

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str, default=model['path'],
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
    parser.add_argument("-f", "--disable_video_file", required=False, action="store_true",
                        help="Disable the output video file creation")
    parser.add_argument("-x", "--disable_video_output", required=False, action="store_true",
                        help="Disable the output of key video frames to stdout\n"
                            "If enabled, you need to pipe the output of this script to ffmpeg\n"
                            f"e.g.: python3 {__file__} | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 1280x720 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm")
    parser.add_argument("-q", "--disable_mqtt", required=False, action="store_true",
                        help="Disable the connection to MQTT server (for example, to test the inference only)")
    parser.add_argument("-s", "--show_window", required=False, action="store_true",
                        help="Shows a Window with the processed output of the image or video")
    parser.add_argument("-k", "--skip_frames", type=int, default=0,
                        help="Skip # of frames on the start of the video.")
    parser.add_argument("-L", "--logfile", type=str, default=DEFAULT_LOGFILE,
                        help="Path to the file to write the log")
    parser.add_argument("-ll", "--loglevel", type=str, default=DEFAULT_LOGLEVEL,
                        help="Level of verbosity log")
    parser.add_argument("--dev", required=False, action="store_true",
                        help="Set options to ease the development.\n"
                        "Same as using -x -s -k 58 -q")
    parser.add_argument("-a", "--all_models", required=False, action="store_true",
                        help="Run all the models in sequence, saving an output video for everyone\n"
                            "For testing and comparation purposes. Implies --dev")
    return parser


def connect_mqtt():
    if USE_MQTT:
        client = mqtt.Client()
        try:
            client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
        except ConnectionRefusedError as err:
            print(f"Could not connect to MQTT server at {MQTT_HOST}:{MQTT_PORT}.", file=sys.stderr)
            print("Is the server up?", file=sys.stderr)
            exit(1)
        return client

def disconnect_mqtt(client):
    if USE_MQTT:
        client.disconnect()

def draw_masks(result, frame, v_width, v_height, prob_threshold):
    '''
    Draw bounding boxes onto the frame.
    '''
    nr_people_on_frame = 0
    
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
        if label != model['person-class']:  # We are only interested in Person/Pedestrian/etc class.
            continue

        confidence = box[2]
        if confidence >= prob_threshold:
            xmin = int(box[3] * v_width)
            ymin = int(box[4] * v_height)
            xmax = int(box[5] * v_width)
            ymax = int(box[6] * v_height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, f"{confidence:.4f}",check_boundary(xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), thickness=2)
            nr_people_on_frame += 1
    return frame, nr_people_on_frame

def draw_stats(frame, nr_people, total_people, duration, vid_duration, frame_nr, v_height):
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
    putText(f"Duration(Vid): {vid_duration:.2f}s")
    putText(f"Total: {total_people}")

    y = v_height - 50
    color = (255, 255, 255)
    putText(f"Model: {model['name']}")
    putText(f"Origin: {model['origin']}")
    putText(f"Total time: {time.perf_counter() - ini_time:.2f}")
    
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

def open_video(path):
    """
    Open the video in _path_
    Return OpenCV CaptureObject
    """
    cap = cv2.VideoCapture(path)
    cap.open(path)
    return cap

def get_video_info(cap):
    """
    Returns informations of the video: (width, height, fps, total_frames)
    """
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #OpenCV 3+

    log.debug(f"Total frames: {total_frames}. {width}x{height}@{fps}FPS")
    
    return (width, height, fps, total_frames)

def calc_duration(duration_start, duration_end):
    """
    Calculates the duration of one person in the frame
    """
    if duration_end > duration_start and duration_end > 0 and duration_start > 0:
        duration = duration_end - duration_start
    else:
        duration = 0
    return duration

def avg(lst):
    if len(lst) == 0:
        return 0
    
    suma = 0
    for val in lst:
        suma += val
    return suma / len(lst)

def print_debug(msg, level=log.DEBUG):
    if allow_print:
        print(msg)
    else:
        log.log(level, msg)

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
    log.debug("Loading model")
    network = infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    required_size = (net_input_shape[3], net_input_shape[2])

    ### Handle the input stream ###
    log.debug("Opening video")
    cap = open_video(args.input)

    v_width, v_height, fps, total_frames = get_video_info(cap)
    
    if isImage:
        out_video_writer = None
    else:
        if not args.disable_video_file:
            log.debug("Creating VideoWriter")
            fourCC = cv2.VideoWriter_fourcc(*'mp4v') #try with: 'MJPG', 'XVID', 'MP4V'
            out_video_writer = cv2.VideoWriter(args.output_video, fourCC, fps, (v_width, v_height))

    ### Loop until stream is over ###
    frame_nr=0
    total_people_counted = 0
    previous_nr_people_on_frame = 0
    duration = 0
    duration_start = 0
    duration_end = 0
    
    vid_duration = 0
    vid_duration_start = 0
    vid_duration_end = 0
    vid_duration_avg = 0

    people = {} #{'person_id': 'duration'}
    vid_people = {} #{'person_id': 'vid_duration'}

    while cap.isOpened:
        frame_nr += 1

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
        start_inference_time = time.perf_counter()
        infer_network.exec_net(frame)

        ### Wait for the result ###
        if infer_network.wait() == 0:
            if args.dev:
                print("Inference time: ", str(time.perf_counter() - start_inference_time))
            ### Get the results of the inference request ###
            result = infer_network.get_output() #[1,1,200,7]

            ### Extract any desired stats from the results ###
            out_frame, nr_people_on_frame = draw_masks(result, raw_frame, v_width, v_height, args.prob_threshold)

            ##################################################################################
            ### Calculation of nr of people and duration
            ##################################################################################
            #if nr_people_on_frame is equal, update the duration (needs to be per-person)
            #else, there is someone new or someone less
            exited = False
            if previous_nr_people_on_frame == nr_people_on_frame:
                if duration_start != 0:
                    duration_end = time.perf_counter()  #calculate the total time of the person until this frame
                    vid_duration_end = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            else:
                if previous_nr_people_on_frame < nr_people_on_frame:
                    #new people on frame
                    total_people_counted += nr_people_on_frame - previous_nr_people_on_frame
                    duration_start = time.perf_counter()
                    vid_duration_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                else:
                    #less people on frame
                    duration_end = time.perf_counter()
                    vid_duration_end = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    duration_start = 0
                    exited = True
                    if duration < 1:
                        total_people_counted -= previous_nr_people_on_frame
                        duration_end += duration

            previous_nr_people_on_frame = nr_people_on_frame

            person_stats = {'count': nr_people_on_frame, 'total': total_people_counted}

            avg_duration = 0
            if exited:
                if duration > 1:
                    people[total_people_counted] = duration
                    vid_people[total_people_counted] = vid_duration
                print_debug("###################################################")
                print_debug(people)
                print_debug("###################################################")
                print_debug("avg duration: ")
                avg_duration = avg(people.values())
                print_debug(avg_duration)

                vid_duration_avg = avg(vid_people.values())
                print_debug("avg vid_duration: ")
                print_debug(vid_duration_avg)

            duration = calc_duration(duration_start, duration_end)
            vid_duration = calc_duration(vid_duration_start, vid_duration_end)
            ##################################################################################
            ### End Calculation of nr of people and duration
            ##################################################################################

            log.debug(f"frame: {frame_nr} ###### count: {nr_people_on_frame}, total: {total_people_counted}, duration: {duration}")
            
            out_frame = draw_stats(out_frame, nr_people_on_frame, total_people_counted, duration, vid_duration, frame_nr, v_height)

            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if USE_MQTT:
                mqtt_client.publish("person", json.dumps(person_stats))
                if duration != 0:
                    mqtt_client.publish("person/duration", json.dumps({'duration': vid_duration_avg if vid_duration_avg != 0 else vid_duration }))

        ### Send the frame to the FFMPEG server ###
        if not args.disable_video_output:
            sys.stdout.buffer.write(out_frame)
            sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        if isImage:
            cv2.imwrite('output_image.jpg', out_frame)
        else:
            if not args.disable_video_file:
                out_video_writer.write(out_frame)

        if args.show_window:
            cv2.imshow('display', out_frame)

            key_pressed = cv2.waitKey(3000 if isImage else 30)
            if key_pressed == 27 or key_pressed == 113: #Esc or q
                break #exit the while cap.isOpened() loop

            if key_pressed == 32: #if space: advance 20 frames
                args.skip_frames = frame_nr + 20

    if not isImage:
        if not args.disable_video_file:
            out_video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

def get_youtube_video_url(url):
    videoPafy = pafy.new(url)
    best = videoPafy.getbest() #preftype="webm")
    return best.url

def check_video_or_pic(input):
    """
    Check if input file is a supported image or movie.
    If not, notify the user and abort the program.
    """
    # based on https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
    pics = ["jpg", "jpeg", "png", "gif", "bmp", "jpe", "jp2", "tiff", "tif"]
    movs = ["avi", "mpg", "mp4", "mkv", "ogv"]

    ext = os.path.splitext(input)[1][1:]
    if ext in pics:
        isImage = True
    elif ext in movs:
        isImage = False
    else:
        print("Input file format not supported", file=sys.stderr)
        exit(1)
    
    return isImage

def loop_all_models(args):
    global model
    global ini_time

    models = Model()
    for m in models:
        if not m['enabled']:
            continue

        model = m
        args.model = model['path']
        args.output_video = f"out/{model['name']}-{dt.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        print_init(args)
        ini_time = time.perf_counter()
        infer_on_stream(args, None)


def sanitize_input(args):
    global isImage
    global allow_print
    
    if args.input == "CAM" or args.input == "0":
        args.input = 0 #the webcam
    else:
        if args.input.startswith('https://www.youtube.com'):
            args.input = get_youtube_video_url(args.input)
        elif args.input.startswith('rtsp://'):
            pass #accept rtsp
        else:
            isImage = check_video_or_pic(args.input) # exits if invalid file type
            args.input = os.path.abspath(args.input)

    if args.all_models:
        args.dev = True

    if args.dev:
        args.disable_video_output = True
        args.show_window = True
        args.disable_mqtt = True
        if args.input == os.path.abspath(DEFAULT_INPUT) and args.skip_frames == 0:
            args.skip_frames = 58

    if args.disable_video_output:
        allow_print = True

    log.basicConfig(filename=args.logfile, level=args.loglevel)

def print_init(args):
    """
    Logs a message to indicate a new run.
    """
    log.info("\n######################")
    log.info(f"Model: {args.model}, input: {args.input}.")

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    global USE_MQTT

    # Grab command line args
    args = build_argparser().parse_args()
    sanitize_input(args)

    if args.all_models:
        loop_all_models(args)
    else:
        print_init(args)

        # Connect to the MQTT server
        USE_MQTT = not args.disable_mqtt
        mqtt_client = connect_mqtt()
        # Perform inference on the input stream
        infer_on_stream(args, mqtt_client)
        disconnect_mqtt(mqtt_client)


if __name__ == '__main__':
    main()
