#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):

    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:].lower() in ['.mp4', '.mov']:
        video_out = image_path[:-4] + '_detected' + image_path[-4:]

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Frame w/h: {}/{}".format(frame_w, frame_h))

        if frame_h > frame_w:
            x_start = 0
            x_end = frame_w - 1
            y_start = int((frame_h - frame_w) / 2)
            y_end = frame_h - y_start - 1
        else:
            y_start = 0
            y_end = frame_h - 1
            x_start = int((frame_w - frame_h) / 2)
            x_end = frame_w - x_start - 1


        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                       15.0,
                                       (x_end - x_start, y_end - y_start))

        print("x_start: {}".format(x_start))
        print("x_end: {}".format(x_end))
        print("y_start: {}".format(y_start))
        print("y_end: {}".format(y_end))
        for i in tqdm(range(nb_frames)):
            ret_val, image = video_reader.read()
            if ret_val:
                cropped = image[y_start:y_end, x_start:x_end]
                boxes = yolo.predict(cropped, obj_threshold=0.75)
                final_image = draw_boxes(cropped, boxes, config['model']['labels'])
                video_writer.write(np.uint8(final_image))
            else:
                print("No frame read")

        video_reader.release()
        video_writer.release()  
    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
