import os
from functools import partial
from logging import getLogger
from typing import List

import cv2
import numpy as np
import torch
from fastai.metrics import dice
from fastai.vision import Image

from video699.quadrangle.geos import GEOSConvexQuadrangle
from video699.video.annotated import AnnotatedSampledVideoScreenDetector
from shutil import copyfile

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


def create_labels(videos, labels_path):
    actual_detector = AnnotatedSampledVideoScreenDetector()
    if not labels_path.absolute().exists():
        os.mkdir(labels_path.absolute())

    for video in videos:
        video_dir = labels_path / video.filename
        if not video_dir.exists():
            os.mkdir(video_dir.absolute())

        for frame in video:
            frame_path = video_dir / frame.filename
            if not frame_path.exists():
                mask = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
                screens = actual_detector.detect(frame=frame)
                for screen in screens:
                    points = (
                        screen.coordinates.top_left,
                        screen.coordinates.top_right,
                        screen.coordinates.bottom_right,
                        screen.coordinates.bottom_left
                    )
                    cv2.fillConvexPoly(mask,
                                       np.array([[[xi, yi]] for xi, yi in points]).astype(np.int32),
                                       (1, 1, 1))

                mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(str(frame_path.absolute()), mask_gray)


def create_images(videos, images_path):
    if not images_path.absolute().exists():
        os.mkdir(images_path.absolute())

    for video in videos:
        video_dir = images_path / video.filename
        if not video_dir.exists():
            os.mkdir(video_dir.absolute())

        for frame in video:
            frame_path = video_dir / frame.filename
            if not frame_path.exists():
                copyfile(frame.pathname, str(frame_path.absolute()))


def parse_factors(factors_string: str) -> List[float]:
    return list(map(float, factors_string.split(', ')))


def parse_lr(lr_string):
    lrs = list(map(float, lr_string.split(', ')))
    if len(lrs) == 1:
        lr = lrs[0]
    elif len(lrs) == 2:
        lr = slice(lrs[0], lrs[1])
    else:
        LOGGER.error("Format of learning rate not understood. Using default 1e-03.")
        lr = 1e-03
    return lr


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
                               'factors': parse_factors(config['base_factors']),
                               }
        except KeyError as ex:
            LOGGER.error(f"{ex.__traceback__}")
            LOGGER.error(f"Required parameter does not exists in default.ini")

    if erode_dilate:
        try:
            methods['erode_dilate'] = {'lower_bound': config.getint('erode_dilate_lower_bound'),
                                       'upper_bound': config.getint('erode_dilate_upper_bound'),
                                       'factors': parse_factors(config['erode_dilate_factors']),
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


def iou(pred, actual):
    return dice(pred, actual, iou=True)
