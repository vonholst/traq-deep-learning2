#! /usr/bin/env python

#from frontend import YOLO
import json
import keras
from keras.models import Model, Input
from keras.applications import MobileNet
from keras.layers import Conv2D, Reshape
import coremltools
from backend import MobileNetFeature, MOBILENET_BACKEND_PATH, FullYoloFeature, FULL_YOLO_BACKEND_PATH
import argparse

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
    coreml_model.short_description = 'Model for tracking toys: {}'.format(json.dumps(labels))
    coreml_model.author = 'Per Default'
    coreml_model.license = 'BSD'
    coreml_model.save(saved_model_name)
