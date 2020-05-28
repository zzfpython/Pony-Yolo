"""Miscellaneous utility functions."""

from datetime import datetime
import colorsys
import os
import sys
from functools import reduce
from functools import wraps

import math
import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, Activation, add, Lambda, concatenate, MaxPooling2D
from keras.layers import Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adadelta, Adagrad
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

np.set_printoptions(precision=3, suppress=True)
MAX_VERTICES = 1000
ANGLE_STEP  = 15
NUM_ANGLES3  = int(360 // ANGLE_STEP * 3)
NUM_ANGLES  = int(360 // ANGLE_STEP)

grid_size_multiplier = 8
anchor_mask = [[0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8]]
anchors_per_level = 9

dropped_boxes = 0
used_boxes = 1


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw = image.shape[1]
    ih = image.shape[0]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    cvi = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cvi = cv.resize(cvi, (nw, nh), interpolation=cv.INTER_CUBIC)
    dx = int((w - nw) // 2)
    dy = int((h - nh) // 2)
    new_image = np.zeros((h, w, 3), dtype='uint8')
    new_image[...] = 128
    if nw <= w and nh <= h:
        new_image[dy:dy + nh, dx:dx + nw, :] = cvi
    else:
        new_image = cvi[-dy:-dy + h, -dx:-dx + w, :]

    return new_image.astype('float32') / 255.0


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=80, jitter=.1, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    #print(annotation_line)
    line = annotation_line
    #line = annotation_line.split()

    #for element in range(1, len(line)):
    #    # TODO: Udelat check na pripad, kdy je v souboru vice vrcholu, nez mame max vertices
    #    for symbol in range(line[element].count(',') - 4, MAX_VERTICES * 2, 2):
    #        line[element] = line[element] + ',0,0'

    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    # rozseka radek v label textaku prve na boxy a potom samotne hodnoty boxu podle carky
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        #box_data = np.zeros((max_boxes, 5))
        box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box), 0:5] = box[:, 0:5]

            for b in range(0, len(box)):
                for i in range(5, MAX_VERTICES * 2, 2):
                    if box[b,i] == 0 and box[b, i + 1] == 0:
                        continue
                    box[b, i] = box[b, i] * scale + dx
                    box[b, i + 1] = box[b, i + 1] * scale + dy

            box_data[:, i:NUM_ANGLES3 + 5] = 0
            #for i in range(5, NUM_ANGLES3 + 5, 3):
            #    box_data[:, i] = 0  # vzdalenost
            #    box_data[:, i + 1] = 0  # uhel
            #    box_data[:, i + 2] = 0  # confidence

            for i in range(0, len(box)):
                boxes_xy = (box[i, 0:2] + box[i, 2:4]) // 2
                boxes_wh = (box[i, 2:4] - box[i, 0:2])

                for ver in range(5, MAX_VERTICES * 2, 2):  # zde je inkrementace o 2, protoze se zpracovavaji data z annotation line, jeste neexpandovana
                    if box[i, ver] == 0 and box[i, ver + 1] == 0:
                        break
                    dist_x = boxes_xy[0] - box[i, ver]  # x vzdalenost vrcholu polygonu od stredu boxu, absolutni vzdalenost
                    dist_y = boxes_xy[1] - box[i, ver + 1]  # y vzdalenost vrcholu polygonu od stredu boxu, absolutni vzdalenost
                    dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))  # mame vzdalenost
                    if (dist < 1): dist = 1

                    angle = np.degrees(np.arctan2(dist_y, dist_x))
                    if (angle < 0): angle += 360
                    iangle = int(angle) // ANGLE_STEP
                    relative_angle = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP

                    if dist > box_data[i, 5 + iangle * 3]:  # koukame, jestli uz existuje a bereme ten vzdalenejsi vrchol
                        box_data[i, 5 + iangle * 3] = dist
                        box_data[i, 5 + iangle * 3 + 1] = relative_angle
                        box_data[i, 5 + iangle * 3 + 2] = 1
        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    #new_ar = 1.0
    scale = rand(.6, 1.8)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    nwiw = nw/iw
    nhih = nh/ih

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1


    # correct boxes
    box_data = np.zeros((max_boxes, 5 + NUM_ANGLES3))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nwiw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nhih + dy
        if flip: box[:, [0, 2]] = (w-1) - box[:, [2, 0]]

        for b in range(0, len(box)):
            for i in range(5, MAX_VERTICES * 2, 2):
                if box[b, i] == 0 and box[b, i + 1] == 0:
                    continue
                box[b, i] = np.clip(box[b, i] * nwiw + dx, 0, w-1)
                box[b, i + 1] = np.clip(box[b, i + 1] * nhih + dy, 0, h-1)
                if flip:
                    box[b, i] = (w - 1) - box[b, i]

        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] >= w] = w-1
        box[:, 3][box[:, 3] >= h] = h-1
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box), 0:5] = box[:, 0:5]

        box_data[:, i:NUM_ANGLES3 + 5] = 0

        for i in range(0, len(box)):
            boxes_xy = (box[i, 0:2] + box[i, 2:4]) // 2

            for ver in range(5, MAX_VERTICES * 2, 2):  # zde je inkrementace o 2, protoze se zpracovavaji data z annotation line, jeste neexpandovana
                if box[i, ver] == 0 and box[i, ver + 1] == 0:
                    break
                dist_x = boxes_xy[0] - box[i, ver]  # x vzdalenost vrcholu polygonu od stredu boxu, absolutni vzdalenost
                dist_y = boxes_xy[1] - box[i, ver + 1]  # y vzdalenost vrcholu polygonu od stredu boxu, absolutni vzdalenost
                dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))  # mame vzdalenost
                if (dist < 1): dist = 1

                angle = np.degrees(np.arctan2(dist_y, dist_x))
                if (angle < 0): angle += 360
                iangle = int(angle) // ANGLE_STEP
                if iangle==NUM_ANGLES: iangle = 0

                if dist > box_data[i, 5 + iangle * 3]: #koukame, jestli uz existuje a bereme ten vzdalenejsi vrchol
                    box_data[i, 5 + iangle * 3]     = dist
                    box_data[i, 5 + iangle * 3 + 1] = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP #relative angle
                    box_data[i, 5 + iangle * 3 + 2] = 1

    return image_data, box_data


"""YOLO_v3 Model Defined in Keras."""


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        y = squeeze_excite_block(y)
        x = Add()([x, y])
    return x


# https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se_resnet.py
def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = LeakyReLU(alpha=0.1)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')


def _resnet_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block without bottleneck layers
    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer
    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = LeakyReLU(alpha=0.1)(x)
    # x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    m = LeakyReLU()(m)
    return m

def resnetblock_body(x, num_filters, num_blocks, strides=(1, 1)):
    x = _resnet_block(x, num_filters, strides=strides)
    for i in range(1, num_blocks):
        x = _resnet_block(x, num_filters)
    return x


def resnet_body(x):
    base = 8

    # ResNet - 50 = [3, 4, 6, 3]
    # ResNet - 101 = [3, 6, 23, 3]
    # ResNet - 152 = [3, 8, 36, 3]

    #initial block
    x = Conv2D(base * 8, (7, 7), padding='same', use_bias=False, strides=(2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    #projection block
    x = resnetblock_body(x, base * 8, 3)

    #pooling blocks
    x = resnetblock_body(x, base * 16, 4, (2, 2))
    big = x
    x = resnetblock_body(x, base * 32, 6, (2, 2))
    medium = x
    x = resnetblock_body(x, base * 64, 3, (2, 2))
    small = x
    return small, medium, big


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    base = 4  # orig base = 8
    x = DarknetConv2D_BN_Leaky(base * 4, (3, 3))(x)
    x = resblock_body(x, base * 8, 1)
    x = resblock_body(x, base * 16, 2)
    x = resblock_body(x, base * 32, 8)
    small = x
    x = resblock_body(x, base * 64, 8)
    medium = x
    x = resblock_body(x, base * 128, 8)
    big = x
    return small, medium, big



def make_last_layers(x, num_filters, out_filters):
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    small, medium, big = darknet_body(inputs)

    base = 4
    small  = DarknetConv2D_BN_Leaky(base*32, (1, 1))(small)
    medium = DarknetConv2D_BN_Leaky(base*32, (1, 1))(medium)
    big    = DarknetConv2D_BN_Leaky(base*32, (1, 1))(big)

    all = Add()([medium, UpSampling2D(2,interpolation='bilinear')(big)])
    all = Add()([small, UpSampling2D(2,interpolation='bilinear')(all)])



    num_filters = base*32

    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(all)

    all = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5 + NUM_ANGLES3), (1, 1)))(x)

    return Model(inputs, all)



def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = anchors_per_level
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(tf.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1], name='yolo_head/tile/reshape/grid_y'),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(tf.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1], name='yolo_head/tile/reshape/grid_x'),
                    [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1, name='yolo_head/concatenate/grid')
    grid = K.cast(grid, K.dtype(feats))
    global _var
    _var = [grid_shape, feats, anchors_tensor]
    feats = tf.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5 + NUM_ANGLES3], name='yolo_head/reshape/feats')

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))

    box_confidence      = K.sigmoid(feats[..., 4:5])
    box_class_probs     = K.sigmoid(feats[..., 5:5 + num_classes])
    polygons_confidence = K.sigmoid(feats[..., 5+num_classes+2:5+num_classes+NUM_ANGLES3:3])
    # zapsani vrcholu polygonu
    polygons_x = K.exp(feats[..., 5 + num_classes:num_classes + 5 + NUM_ANGLES3:3])

    dx = K.square(anchors_tensor[..., 0:1] / 2)
    dy = K.square(anchors_tensor[..., 1:2] / 2)
    d = K.cast(K.sqrt(dx + dy), K.dtype(polygons_x))
    a = K.pow(input_shape[::-1], 2)
    a = K.cast(a, K.dtype(feats))
    b= K.sum(a)
    diagonal =  K.cast(K.sqrt(b), K.dtype(feats))
    polygons_x = polygons_x * d / diagonal

    polygons_y = feats[..., 5 + num_classes + 1:num_classes + 5 + NUM_ANGLES3:3]
    polygons_y = K.sigmoid(polygons_y)

    if calc_loss == True:
        return grid, feats, box_xy, box_wh, polygons_confidence
    return box_xy, box_wh, box_confidence, box_class_probs, polygons_x, polygons_y, polygons_confidence


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_correct_polygons(polygons_x, polygons_y, polygons_confidence, boxes, input_shape, image_shape):
    #diagonal = K.sqrt(K.pow(image_shape[0], 2) + K.pow(image_shape[1], 2))
    #polygons_x *= diagonal

    polygons = K.concatenate([polygons_x, polygons_y, polygons_confidence])
    return polygons


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs, polygons_x, polygons_y, polygons_confidence = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    polygons = yolo_correct_polygons(polygons_x, polygons_y, polygons_confidence, boxes, input_shape, image_shape)
    polygons = K.reshape(polygons, [-1, NUM_ANGLES3])
    return boxes, box_scores, polygons


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=80,
              score_threshold=.5,
              iou_threshold=.4):
    """Evaluate YOLO model on given input and return filtered boxes."""
    input_shape = K.shape(yolo_outputs)[1:3] * grid_size_multiplier
    boxes = []
    box_scores = []
    polygons = []

    for l in range(1):
        _boxes, _box_scores, _polygons = yolo_boxes_and_scores(yolo_outputs,
                                                               anchors[anchor_mask[l]], num_classes, input_shape,
                                                               image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
        polygons.append(_polygons)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    polygons = K.concatenate(polygons, axis=0)

    mask = box_scores >= score_threshold
    box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    polygons_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_polygons = tf.boolean_mask(polygons, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        class_polygons = K.gather(class_polygons, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
        polygons_.append(class_polygons)
    polygons_ = K.concatenate(polygons_, axis=0)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)



    return boxes_, scores_, classes_, polygons_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5+69)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
        vstup je to nase kratke
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    # shape = [2, 25, 85] == [batch, max_boxes, delka vektoru] - tady je asi problem -> u kazdeho z 25 boxu mame ten polygon, tedy 25x duplicita
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # tady jsou uz stredy
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]


    # vydeli vrcholy polygonu velikosti obrazku -> relativni souradnice
    #print('preprocess true boxes------------------')
    #TODO: toto prepsat pres slicy, at to je krute rychlejsi
    #for b in range(true_boxes.shape[0]): #z absolutni vzdalenosti vypoctene v get_random_data udelame relativni
    #    for boxes in range(true_boxes.shape[1]):
    #        true_boxes[b, boxes, 5:NUM_ANGLES3 + 5:3] /= np.sqrt(np.power(boxes_wh[b, boxes, 0], 2) + np.power(boxes_wh[b, boxes, 1], 2))

    #print(true_boxes[:,:, 5:NUM_ANGLES3 + 5:3].shape)
    #print(boxes_wh.shape)
    #print(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)).shape)


    true_boxes[:,:, 5:NUM_ANGLES3 + 5:3] /= np.clip(np.expand_dims(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)), -1), 0.0001, 9999999)
    #true_boxes[..., 5: NUM_ANGLES3 + 5:3] /= np.sqrt(np.power(boxes_wh[..., 0],2) + np.power(boxes_wh[..., 1],2))
    #print(true_boxes)

    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: grid_size_multiplier}[l] for l in range(1)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes + NUM_ANGLES3),
                       dtype='float32') for l in range(1)]


    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0
    # pro vsechny boxy

    global dropped_boxes
    global used_boxes

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            l = 0
            if n in anchor_mask[l]:
                #try:
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[b, t, 4].astype('int32')

                #if y_true[l][b, j, i, k, 4] == 1:
                #    dropped_boxes += 1
                #else:
                #    used_boxes += 1

                y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                y_true[l][b, j, i, k, 4] = 1
                y_true[l][b, j, i, k, 5 + c] = 1
                y_true[l][b, j, i, k, 5 + num_classes:5 + num_classes + NUM_ANGLES3] = true_boxes[b, t, 5: 5 + NUM_ANGLES3]
    #print(' ', dropped_boxes, used_boxes, dropped_boxes/used_boxes*100.0)
    return y_true


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = 1
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    g_y_true = y_true
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * grid_size_multiplier, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0

    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    for layer in range(num_layers):
        object_mask = y_true[layer][..., 4:5]
        vertices_mask = y_true[layer][..., 5 + num_classes + 2:5 + num_classes + NUM_ANGLES3:3]
        true_class_probs = y_true[layer][..., 5:5 + num_classes]

        grid, raw_pred, pred_xy, pred_wh, pol_cnf = yolo_head(yolo_outputs[layer], anchors[anchor_mask[layer]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        # grid_shape je list ktery drzi dimenze mrizky pro vsechna rozliseni, kazdy prvek listu je numpy array o dvou prvcich

        # grid jsou indexy vsech bunek (0,1) (0,2) (0,3) ... (51, 51) atd.

        # Darknet raw box to calculate loss.
        # stred boxu z [0, 1] * [velikost gridu] - [index bunky v gridu]
        raw_true_xy = y_true[layer][..., :2] * grid_shapes[layer][::-1] - grid
        raw_true_polygon0 = y_true[layer][..., 5 + num_classes: 5 + num_classes + NUM_ANGLES3]

        raw_true_wh = K.log(y_true[layer][..., 2:4] / anchors[anchor_mask[layer]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf

        raw_true_polygon_x = raw_true_polygon0[..., ::3]
        raw_true_polygon_y = raw_true_polygon0[..., 1::3]

        dx = K.square(anchors[anchor_mask[layer]][..., 0:1] / 2)
        dy = K.square(anchors[anchor_mask[layer]][..., 1:2] / 2)
        d = K.cast(K.sqrt(dx + dy), K.dtype(raw_true_polygon_x))
        # raw_true_polygon_x = K.log(raw_true_polygon_x / anchors[anchor_mask[l]][...,0:1] * input_shape[::-1][0])

        diagonal = K.sqrt(K.pow(input_shape[::-1][0], 2) + K.pow(input_shape[::-1][1], 2))
        raw_true_polygon_x = K.log(raw_true_polygon_x / d * diagonal)
        raw_true_polygon_x = K.switch(vertices_mask, raw_true_polygon_x, K.zeros_like(raw_true_polygon_x))
        box_loss_scale = 2 - y_true[layer][..., 2:3] * y_true[layer][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[layer][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                                                                                                             from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:5 + num_classes], from_logits=True)
        polygon_loss_x = object_mask * vertices_mask * box_loss_scale * 0.5 * K.square(raw_true_polygon_x - raw_pred[..., 5 + num_classes:5 + num_classes + NUM_ANGLES3:3])
        polygon_loss_y = object_mask * vertices_mask * box_loss_scale * K.binary_crossentropy(raw_true_polygon_y, raw_pred[..., 5 + num_classes + 1:5 + num_classes + NUM_ANGLES3:3], from_logits=True)
        vertices_confidence_loss = object_mask * K.binary_crossentropy(vertices_mask, raw_pred[..., 5 + num_classes + 2:5 + num_classes + NUM_ANGLES3:3], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        class_loss = K.sum(class_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        vertices_confidence_loss = K.sum(vertices_confidence_loss) / mf
        polygon_loss = K.sum(polygon_loss_x) / mf + K.sum(polygon_loss_y) / mf
        #r_diou = K.sum(diou) / mf

        loss += (xy_loss + wh_loss + confidence_loss + class_loss + 0.2 * polygon_loss + 0.2 * vertices_confidence_loss)/ (K.sum(object_mask) + 1)
    return loss


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'yolo_classes.txt',
        "score": 0.1,
        "iou": 0.2,
        "model_image_size": (320,608),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes, self.polygons = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), anchors_per_level, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            # novy output
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5 + NUM_ANGLES3), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes, polygons = yolo_eval(self.yolo_model.output, self.anchors,
                                                     len(self.class_names), self.input_image_shape,
                                                     score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes, polygons

    def detect_image(self, image):
        # start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            print('THE functionality is not implemented!')
            # new_image_size = (image.width - (image.width % 32),
            #                  image.height - (image.height % 32))
            # boxed_image = letterbox_image(image, new_image_size)
        # image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        # image_data /= 255.
        image_data = np.expand_dims(boxed_image, 0)  # Add batch dimension.
        #print('image data shape', image_data.shape)
        #print('image size', image.size)

        out_boxes, out_scores, out_classes, polygons = self.sess.run(
            [self.boxes, self.scores, self.classes, self.polygons],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
        '''
        global _var
        #_var = [grid_x, grid_y, feats, anchors_tensor]
        output = self.sess.run(
            [_var],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        output = output[0]
        print(output[0], output[1].shape, output[2].shape)
        '''

        for b in range(0, out_boxes.shape[0]):
            cy = (out_boxes[b, 0] + out_boxes[b, 2]) // 2
            cx = (out_boxes[b, 1] + out_boxes[b, 3]) // 2
            diagonal = np.sqrt(np.power(out_boxes[b, 3] - out_boxes[b, 1], 2.0) + np.power(out_boxes[b, 2] - out_boxes[b, 0], 2.0))
            for i in range(0, NUM_ANGLES):
                x1 = cx - math.cos(math.radians((polygons[b, i+NUM_ANGLES] + i) / NUM_ANGLES * 360)) * polygons[b, i] *diagonal# scale[1]
                y1 = cy - math.sin(math.radians((polygons[b, i+NUM_ANGLES] + i) / NUM_ANGLES * 360)) * polygons[b, i] *diagonal# scale[0]
                polygons[b, i]            = x1# / scale[0] - offset[0]
                polygons[b, i+NUM_ANGLES] = y1# / scale[1] - offset[1]


        return out_boxes, out_scores, out_classes, polygons

    def close_session(self):
        self.sess.close()


if __name__ == "__main__":

    """
    Retrain the YOLO model for your own dataset.
    """


    def _main():
        phase = 1
        annotation_path = 'train.txt'
        validation_path = 'val.txt'
        log_dir = 'detector_lite/'
        classes_path = 'yolo_classes.txt'
        anchors_path = 'yolo_anchors.txt'
        class_names = get_classes(classes_path)
        num_classes = len(class_names)
        anchors = get_anchors(anchors_path)

        input_shape = (320,608) # multiple of 32, hw

        if phase == 1:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
        else:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=True, weights_path=log_dir+'pretrained_model.h5')


        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=1, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, delta=0.03)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        with open(annotation_path) as f:
            lines = f.readlines()

        with open(validation_path) as f:
            lines_val = f.readlines()

        for i in range (0, len(lines)):
            lines[i] = lines[i].split()
            for element in range(1, len(lines[i])):
                for symbol in range(lines[i][element].count(',') - 4, MAX_VERTICES * 2, 2):
                    lines[i][element] = lines[i][element] + ',0,0'

        for i in range(0, len(lines_val)):
            lines_val[i] = lines_val[i].split()
            for element in range(1, len(lines_val[i])):
                for symbol in range(lines_val[i][element].count(',') - 4, MAX_VERTICES * 2, 2):
                    lines_val[i][element] = lines_val[i][element] + ',0,0'

        num_val = int(len(lines_val))
        num_train = len(lines)



        batch_size = 8
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))



        model.compile(optimizer=Adadelta(1.0), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        epochs = 100
        history = model.fit_generator(data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes),
                                      steps_per_epoch=max(1, num_train // batch_size),
                                      validation_data=data_generator_wrapper(lines_val, batch_size, input_shape, anchors, num_classes),
                                      validation_steps=max(1, num_val // batch_size),
                                      epochs=epochs,
                                      initial_epoch=0,
                                      callbacks=[reduce_lr, early_stopping, checkpoint])



    def get_classes(classes_path):
        """loads the classes"""
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                     weights_path='model_data/yolo_weights.h5'):
        """create the training model"""
        K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        num_anchors = len(anchors)
        y_true = Input(shape=(h // grid_size_multiplier, w // grid_size_multiplier, anchors_per_level, num_classes + 5 + NUM_ANGLES3))

        model_body = yolo_body(image_input, anchors_per_level, num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [model_body.output, y_true])
        model = Model([model_body.input, y_true], model_loss)

        # print(model.summary())
        return model


    def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
        """data generator for fit_generator"""
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                #print(i)
                # image, box = get_random_data(annotation_lines[i], input_shape, random=False)
                # turned off preprocessing because image will be disorted
                image, box = get_random_data(annotation_lines[i], input_shape, random=True, jitter=0, hue=0)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)


    def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0: return None
        return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


    if __name__ == '__main__':
        _main()
