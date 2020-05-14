# -*- coding: utf-8 -*-

"""
This module serves as common utility function storage for postprocessing and fastai_detector.
"""
import os
from logging import getLogger
from shutil import copyfile
from typing import List

import cv2
import numpy as np
import torch
from fastai.metrics import dice
from fastai.vision import Image

from video699.configuration import get_configuration
from video699.video.annotated import AnnotatedSampledVideoScreenDetector

LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['FastAIScreenDetector']
image_area = CONFIGURATION.getint('image_width') * CONFIGURATION.getint('image_height')


def cv_image_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    tensor = Image(tensor.to(torch.float32) / 255)
    return tensor


def tensor_to_cv_binary_image(tensor):
    return np.squeeze(np.transpose(tensor[1].numpy(), (1, 2, 0))).astype('uint8')


def resize_pred(pred, width, height):
    predicted_resized = cv2.resize(pred, dsize=(width, height))
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
        lr = slice(lrs[0])
    elif len(lrs) == 2:
        lr = slice(lrs[0], lrs[1])
    else:
        LOGGER.error("Format of learning rate not understood. Using default 1e-03.")
        lr = 1e-03
    return lr


def parse_train_params(config):
    train_params = {}
    for param in ['batch_size', 'resize_factor', 'frozen_epochs',
                  'unfrozen_epochs', 'frozen_lr', 'unfrozen_lr']:
        train_params[param] = parse_lr(config[param]) if '_lr' in param else config.getint(param)
    return train_params


def parse_post_processing_params(config):
    base = config.getboolean('base')
    erosion_dilation = config.getboolean('erosion_dilation')
    ratio_split = config.getboolean('ratio_split')

    if not (base or erosion_dilation or ratio_split):
        LOGGER.error("No valid arguments provided in default.ini")

    post_processing_params = {'base': base, 'erosion_dilation': erosion_dilation, 'ratio_split': ratio_split}
    try:
        if base:
            post_processing_params.update({'base_lower_bound': config.getint('base_lower_bound'),
                                           'base_factors': parse_factors(config['base_factors']),
                                           })

        if erosion_dilation:
            post_processing_params.update(
                {'erosion_dilation_lower_bound': config.getint('erosion_dilation_lower_bound'),
                 'erosion_dilation_factors': parse_factors(config['erosion_dilation_factors']),
                 'erosion_dilation_kernel_size': config.getint('erosion_dilation_kernel_size'),
                 })

        if ratio_split:
            post_processing_params.update(
                {'ratio_split_lower_bound': config.getfloat('ratio_split_lower_bound'),
                 })

    except KeyError as ex:
        LOGGER.error(f"{ex.__traceback__}")
        LOGGER.error(f"Required parameter does not exists in default.ini")
    return post_processing_params


class NotFittedException(Exception):
    pass


def acc(pred, actual):
    """FastAI semantic segmentation binary accuracy"""
    actual = actual.squeeze(1)
    return (pred.argmax(dim=1) == actual).float().mean()


def iou_sem_seg(pred, actual):
    return dice(pred, actual, iou=True)


def get_label_from_image_name(labels_output_path, fname):
    return os.path.join(labels_output_path, os.path.join(*str(fname).split('/')[-2:]))


def midpoint(pointA, pointB):
    return (pointA[0] + pointB[0]) / 2, (pointA[1] + pointB[1]) / 2


def get_coordinates(quadrangle):
    squeezed = quadrangle.squeeze()
    x = squeezed[:, 0]
    y = squeezed[:, 1]
    top_left = (x + y).argmin()
    top_right = (max(y) - y + x).argmax()
    bottom_left = (max(x) - x + y).argmax()
    bottom_right = (x + y).argmax()
    return {'top_left': squeezed[top_left],
            'top_right': squeezed[top_right],
            'bottom_right': squeezed[bottom_right],
            'bottom_left': squeezed[bottom_left]}


def draw_polygon(polygon, image):
    copy = image.copy()
    return cv2.fillConvexPoly(copy, polygon, 100)


def iou(screenA, screenB):
    intersection = screenA.coordinates.intersection_area(screenB.coordinates)
    union = screenA.coordinates.union_area(screenB.coordinates)
    return intersection / union


def is_bigger_than_boundary(contour_area, lower_area_percentage):
    contour_percentage = contour_area * 100 / image_area
    return lower_area_percentage < contour_percentage
