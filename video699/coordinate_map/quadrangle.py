# -*- coding: utf-8 -*-

"""This module implements a coordinate map between a video frame and projection screen coordinate
systems based on a quadrangle specifying the screen corners in the video frame coordinate system.

"""

from math import sqrt

import cv2 as cv
import numpy as np

from ..configuration import CONFIGURATION
from ..interface import CoordinateMapABC


BORDER_MODE = cv.__dict__[CONFIGURATION['Quadrangle']['border_mode']]


class Quadrangle(CoordinateMapABC):
    """A map between a video frame and projection screen coordinate systems based on a quadrangle.

    The quadrangle specifies the screen corners in a video frame coordinate system.

    Parameters
    ----------
    top_left : (scalar, scalar)
        The top left corner of the quadrangle in a video frame coordinate system.
    top_right : (scalar, scalar)
        The top right corner of the quadrangle in a video frame coordinate system.
    btm_left : (scalar, scalar)
        The bottom left corner of the quadrangle in a video frame coordinate system.
    btm_right : (scalar, scalar)
        The bottom right corner of the quadrangle in a video frame coordinate system.

    Attributes
    ----------
    width : int
        The width of the screen in a screen coordinate system.
    height : int
        The height of the screen in a screen coordinate system.
    transform : 3 x 3 ndarray
        The perspective transform matrix from a frame coordinate system to a screen coordinate
        system in Homogeneous coordinates.
    """

    def __init__(self, top_left, top_right, btm_left, btm_right):
        top_width = sqrt(((btm_left[0] - btm_right[0])**2) + ((btm_left[1] - btm_right[1])**2))
        btm_width = sqrt(((top_left[0] - top_right[0])**2) + ((top_left[1] - top_right[1])**2))
        left_height = sqrt(((top_left[0] - btm_left[0])**2) + ((top_left[1] - btm_left[1])**2))
        right_height = sqrt(((top_right[0] - btm_right[0])**2) + ((top_right[1] - btm_right[1])**2))
        max_width = max(int(top_width), int(btm_width))
        max_height = max(int(left_height), int(right_height))
        self._width = max_width
        self._height = max_height

        frame_coordinates = np.float32(
            [
                top_left,
                top_right,
                btm_left,
                btm_right,
            ],
        )
        screen_coordinates = np.float32(
            [
                (0, 0),
                (max_width - 1, 0),
                (0, max_height - 1),
                (max_width - 1, max_height - 1),
            ],
        )
        self._transform = cv.getPerspectiveTransform(frame_coordinates, screen_coordinates)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def transform(self):
        return self._transform

    def __call__(self, frame_image):
        return cv.warpPerspective(
            frame_image,
            self.transform,
            (self.width, self.height),
            borderMode=BORDER_MODE,
        )
