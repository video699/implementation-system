import os
from functools import partial
from logging import getLogger

import cv2
from fastai.metrics import dice
import numpy as np
import torch
from fastai.vision import Image

from video699.interface import ScreenABC
from video699.quadrangle.geos import GEOSConvexQuadrangle

LOGGER = getLogger(__name__)


def get_top_left_x(screen: GEOSConvexQuadrangle) -> int:
    return screen.top_left[0]


def cv_image_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    tensor = Image(tensor.to(torch.float32) / 255)
    return tensor


def tensor_to_cv_binary_image(tensor):
    return np.squeeze(np.transpose(tensor[1].numpy(), (1, 2, 0))).astype('uint8')


def resize_pred(pred, new_size):
    predicted_resized = cv2.resize(pred, dsize=new_size)
    return predicted_resized


def parse_methods(config):
    # TODO Re-write
    methods = {}
    base = config.getboolean('base')
    erode_dilate = config.getboolean('erode_dilate')
    ratio_split = config.getboolean('ratio_split')
    if not (base or erode_dilate or ratio_split):
        LOGGER.error("No valid arguments provided in default.ini")

    if base:
        try:
            methods['base'] = {'lower_bound': config.getint('base_lower_bound'),
                               'upper_bound': config.getint('base_upper_bound'),
                               'factors': eval(config['base_factors']),
                               }
        except KeyError as ex:
            LOGGER.error(f"{ex.__traceback__}")
            LOGGER.error(f"Required parameter does not exists in default.ini")

    if erode_dilate:
        try:
            methods['erode_dilate'] = {'lower_bound': config.getint('erode_dilate_lower_bound'),
                                       'upper_bound': config.getint('erode_dilate_upper_bound'),
                                       'factors': eval(config['erode_dilate_factors']),
                                       'iterations': config.getint('erode_dilate_iterations'),
                                       }
        except KeyError as ex:
            LOGGER.error(f"{ex.__traceback__}")
            LOGGER.error(f"Required parameter does not exists in default.ini")

    if ratio_split:
        try:
            methods['ratio_split'] = {'lower_bound': config.getfloat('ratio_split_lower_bound'),
                                      'upper_bound': config.getfloat('ratio_split_upper_bound'),
                                      }
        except KeyError as ex:
            LOGGER.error(f"{ex.__traceback__}")
            LOGGER.error(f"Required parameter does not exists in default.ini")

    return methods


class NotFittedException(Exception):
    pass


def acc(pred, actual):
    """FastAI semantic segmentation binary accuracy"""
    actual = actual.squeeze(1)
    return (pred.argmax(dim=1) == actual).float().mean()


def get_label_from_image_name(labels_output_path, fname):
    # TODO Try to do it through pathlib.Path (maybe all pathing should be created with it.
    return os.path.join(labels_output_path, os.path.join(*str(fname).split('/')[-2:]))


IOU = partial(dice, iou=True)
