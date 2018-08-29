# -*- coding: utf-8 -*-

"""This module implements a coordinate map between a video frame and projection screen coordinate
systems based on a quadrangle specifying the screen corners in the video frame coordinate system.

"""

from math import sqrt

from cv2 import BORDER_REPLICATE, getPerspectiveTransform, warpPerspective
import numpy as np

from ..interface import CoordinateMapABC


class Quadrangle(CoordinateMapABC):
    """A map between a video frame and projection screen coordinate systems based on a quadrangle.

    The quadrangle specifies the screen corners in the video frame coordinate system.

    Parameters
    ----------
    top_left : (scalar, scalar)
        The top left corner of the quadrangle in the video frame coordinate system.
    top_right : (scalar, scalar)
        The top right corner of the quadrangle in the video frame coordinate system.
    btm_left : (scalar, scalar)
        The bottom left corner of the quadrangle in the video frame coordinate system.
    btm_right : (scalar, scalar)
        The bottom right corner of the quadrangle in the video frame coordinate system.

    Attributes
    ----------
    width : int
        The width of the screen in the screen coordinate space.
    height : int
        The height of the screen in the screen coordinate space.
    transform : 3 x 3 ndarray
        The perspective transform matrix from the frame coordinate space to the screen coordinate
        space in Homogeneous coordinates.
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
        self._transform = getPerspectiveTransform(frame_coordinates, screen_coordinates)

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
        return warpPerspective(
            frame_image,
            self.transform,
            (self.width, self.height),
            borderMode=BORDER_REPLICATE,
        )
