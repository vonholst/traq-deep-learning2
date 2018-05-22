from preprocessing import BatchGenerator, parse_annotation
from backend import MobileNetFeature
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

with open("config.json") as config_buffer:
    config = json.loads(config_buffer.read())

train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                            config['train']['train_image_folder'],
                                            config['model']['labels'])
feature_extractor = MobileNetFeature(config['model']['input_size'])

generator_config = {
    'IMAGE_H': config['model']['input_size'],
    'IMAGE_W': config['model']['input_size'],
    'GRID_H': 7,
    'GRID_W': 7,
    'BOX': 5,
    'LABELS': ["volvo_logo", "driver_controls", "pilot_assist"],
    'CLASS': 3,
    'ANCHORS': config['model']['anchors'],
    'BATCH_SIZE': 32,
    'TRUE_BOX_BUFFER': 10,
}

train_batch = BatchGenerator(train_imgs,
                             generator_config,
                             norm=None)

obj_h_list = []
obj_w_list = []

for img in train_imgs:
    h = float(img['height'])
    w = float(img['width'])
    for obj in img['object']:
        x_min = obj['xmin']
        x_max = obj['xmax']
        y_min = obj['ymin']
        y_max = obj['ymax']
        name = obj['name']
        obj_w = float(x_max - x_min)
        obj_h = float(y_max - y_min)
        norm_w = obj_w/w*7  # unit cell
        norm_h = obj_h/h*7  # unit cell
        obj_w_list.append(norm_w)
        obj_h_list.append(norm_h)

plt.figure(1)
plt.plot(obj_w_list, obj_h_list, 'r*')

mean_w = np.mean(obj_w_list)
mean_h = np.mean(obj_h_list)
max_h = np.max(obj_h_list)
max_w = np.max(obj_w_list)
min_h = np.min(obj_h_list)
min_w = np.min(obj_w_list)

print("Max w/h: {max_w}/{max_h}, Min w/h: {min_w}/{min_h}, Mean w/h: {mean_w}/{mean_h}".format(max_w=max_w, max_h=max_h,
                                                                                               min_w=min_w, min_h=min_h,
                                                                                               mean_w=mean_w,
                                                                                               mean_h=mean_h))



plt.figure(2)
batch = train_batch.__getitem__(0)
[x_batch, b_batch], y_batch = batch
for idx in range(32 ):
    print("Showing image {}".format(idx))
    image = x_batch[idx, ...]
    plt.imshow(image/255.0)
    plt.show()
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

