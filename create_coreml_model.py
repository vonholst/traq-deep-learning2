#! /usr/bin/env python

#from frontend import YOLO
import json
import keras
from keras.models import Model, Input
from keras.applications import MobileNet
from keras.layers import Conv2D, Reshape
import coremltools
from backend import MobileNetFeature, MOBILENET_BACKEND_PATH, FullYoloFeature, FULL_YOLO_BACKEND_PATH
import cv2
import numpy as np
from frontend import YOLO
from utils import BoundBox, draw_boxes
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def preprocess_image(image):
    image = image / 255.
    image = image - 0.5
    image = image * 2.
    return image


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = BoundBox(x, y, w, h, confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        #            for i in xrange(len(sorted_indices)):
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    #                    for j in xrange(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if YOLO.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Create coreml model')

    argparser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')

    args = argparser.parse_args()
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    labels = config['model']['labels']
    anchors = config['model']['anchors']
    model_architecture = config['model']['architecture']
    saved_weights_name = config['train']['saved_weights_name']
    saved_model_name = saved_weights_name[:-2] + 'mlmodel'
    model_input_size = config['model']['input_size']

    nbclass = len(labels)

    # custom_objects = {'relu6': keras.applications.mobilenet.relu6,
    #                   'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}

    if model_architecture == "MobileNet":
        input_image = Input(shape=(model_input_size, model_input_size, 3))
        mobilenet = MobileNet(input_shape=(model_input_size, model_input_size, 3), include_top=False)
        mobilenet.load_weights(MOBILENET_BACKEND_PATH)
        x = mobilenet(input_image)
        feature_extractor = Model(input_image, x)
        grid_h, grid_w = feature_extractor.get_output_shape_at(-1)[1:3]

        output = Conv2D(5 * (4 + 1 + nbclass),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='conv_23',
                        kernel_initializer='lecun_normal')(x)
        new_model = Model(input_image, output)
        new_model.load_weights(saved_weights_name)
        coreml_model = coremltools.converters.keras.convert(new_model, input_names='image',
                                                            image_input_names='image',
                                                            image_scale=2./255.,
                                                            red_bias=-1.0,
                                                            green_bias=-1.0,
                                                            blue_bias=-1.0
                                                            )
        coreml_model.short_description = 'Model for tracking specific volvo interior features: {}'.format(json.dumps(labels))

    elif model_architecture == "Full Yolo":
        input_image = Input(shape=(model_input_size, model_input_size, 3))
        yolo_feature = FullYoloFeature(input_size=model_input_size)
        #feature_extractor = yolo_feature.feature_extractor
        grid_h, grid_w = yolo_feature.feature_extractor.get_output_shape_at(-1)[1:3]
        features = yolo_feature.extract(input_image)

        # make the object detection layer
        output = Conv2D(5 * (4 + 1 + nbclass),
                        (1,1), strides=(1,1),
                        padding='same',
                        name='conv_23',
                        kernel_initializer='lecun_normal')(features)
        new_model = Model(input_image, output)
        new_model.summary()
        new_model.load_weights(saved_weights_name)
        coreml_model = coremltools.converters.keras.convert(new_model, input_names='image',
                                                            image_input_names='image',
                                                            image_scale=1./255.,
                                                            )
        coreml_model.short_description = 'Model for tracking specific volvo interior features: {}'.format(
            json.dumps(labels))

    else:
        raise Exception("Unsupported architecture")

    coreml_model.author = 'Per Default'
    coreml_model.license = 'BSD'
    coreml_model.save(saved_model_name)

    if False:
        test_image_path = 'test.jpg'
        test_image = image = cv2.imread(test_image_path)
        image = cv2.resize(test_image, (224, 224))
        # bgr to rgb
        image_rgb = image[:, :, ::-1]

        input_image = np.expand_dims(preprocess_image(image_rgb), 0)

        netout = new_model.predict([input_image])[0]
        new_netout = np.reshape(netout, (7, 7, 5, nbclass + 5))
        boxes = decode_netout(new_netout, anchors, nbclass)
        result_image = draw_boxes(test_image, boxes, config['model']['labels'], r=0, g=255, b=0)
        cv2.imwrite(test_image_path[:-4] + '_keras' + test_image_path[-4:], result_image)

        coreml_image = Image.open(test_image_path)
        coreml_image = coreml_image.resize((224, 224))

        result = coreml_model.predict({'image': coreml_image})
        coreml_netout = result["output1"]
        coreml_netout_mod = coreml_netout.swapaxes(0, 1)
        coreml_netout_mod = coreml_netout_mod.swapaxes(1,2)
        coreml_netout_res = np.reshape(coreml_netout_mod, (7, 7, 5, nbclass + 5))

        coreml_boxes = decode_netout(coreml_netout_res, anchors, nbclass)

        test_image = image = cv2.imread(test_image_path)
        image = cv2.resize(test_image, (224, 224))
        image_rgb = image[:, :, ::-1]

        result_image = draw_boxes(test_image, coreml_boxes, config['model']['labels'], r=255, g=0, b=0)
        cv2.imwrite(test_image_path[:-4] + '_coreml' + test_image_path[-4:], result_image)

