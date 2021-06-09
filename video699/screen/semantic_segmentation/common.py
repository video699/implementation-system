# -*- coding: utf-8 -*-

"""
This module serves as common utility function storage for postprocessing and fastai_detector.
"""
from logging import getLogger
from typing import List

import cv2
import numpy as np
import torch

from video699.configuration import get_configuration

LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['FastAIScreenDetector']
image_area = CONFIGURATION.getint('image_width') * CONFIGURATION.getint('image_height')


class NotFittedException(Exception):
    pass


def midpoint(pointA, pointB):
    return (pointA[0] + pointB[0]) / 2, (pointA[1] + pointB[1]) / 2



def tensor_to_cv_binary_image(tensor):
    return np.squeeze(np.transpose(tensor[1].numpy(), (1, 2, 0))).astype('uint8')


def resize_pred(pred, width, height):
    predicted_resized = cv2.resize(pred, dsize=(width, height))
    return predicted_resized


def draw_polygon(polygon, image):
    copy = image.copy()
    return cv2.fillConvexPoly(copy, polygon, 100)


def is_bigger_than_boundary(contour_area, lower_area_percentage):
    contour_percentage = contour_area * 100 / image_area
    return lower_area_percentage < contour_percentage


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





def parse_factors(factors_string: str) -> List[float]:
    return list(map(float, factors_string.split(', ')))


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
