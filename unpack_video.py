#! /usr/bin/env python

import argparse
import os
import cv2
from tqdm import tqdm

argparser = argparse.ArgumentParser(
    description='unpack images from video')

argparser.add_argument(
    '-in',
    '--input',
    help='path to video-file')

argparser.add_argument(
    '-o',
    '--outdir',
    help='path to output-folder')


def _main_(args):
    out_path = args.outdir
    video_path = args.input

    video_reader = cv2.VideoCapture(video_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    x_start = 0
    x_end = frame_w - 1
    y_start = int((frame_h - frame_w) / 2)
    y_end = frame_h - y_start - 1

    for i in tqdm(range(nb_frames)):
        retval, image = video_reader.read()
        if retval:
            cropped = image[y_start:y_end, x_start:x_end]
            rescaled = cv2.resize(cropped,(400, 400))
            cv2.imwrite('{}/{}image_{}_cropped.jpg'.format(out_path, video_path, i), rescaled)

    video_reader.release()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
