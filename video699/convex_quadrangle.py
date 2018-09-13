# -*- coding: utf-8 -*-

"""This module implements a coordinate map between a video frame and projection screen coordinate
systems based on a quadrangle specifying the screen corners in the video frame coordinate system.

"""

import cv2 as cv
import numpy as np
from shapely.geometry import Point, Polygon

from .interface import ConvexQuadrangleABC


COLOR_TRANSPARENT = (0, 0, 0, 0)


class ConvexQuadrangle(ConvexQuadrangleABC):
    """A convex quadrangle specifying a map between video frame and projection screen coordinates.

    The quadrangle specifies the screen corners in a video frame coordinate system.

    Parameters
    ----------
    top_left : (scalar, scalar)
        The top left corner of the quadrangle in a video frame coordinate system.
    top_right : (scalar, scalar)
        The top right corner of the quadrangle in a video frame coordinate system.
    bottom_left : (scalar, scalar)
        The bottom left corner of the quadrangle in a video frame coordinate system.
    bottom_right : (scalar, scalar)
        The bottom right corner of the quadrangle in a video frame coordinate system.

    Attributes
    ----------
    top_left : (scalar, scalar)
        The top left corner of the quadrangle in a video frame coordinate system.
    top_right : (scalar, scalar)
        The top right corner of the quadrangle in a video frame coordinate system.
    bottom_left : (scalar, scalar)
        The bottom left corner of the quadrangle in a video frame coordinate system.
    bottom_right : (scalar, scalar)
        The bottom right corner of the quadrangle in a video frame coordinate system.
    width : int
        The width of the screen in a screen coordinate system.
    height : int
        The height of the screen in a screen coordinate system.
    """

    def __init__(self, top_left, top_right, bottom_left, bottom_right):
        self._top_left = Point(top_left)
        self._top_right = Point(top_right)
        self._bottom_left = Point(bottom_left)
        self._bottom_right = Point(bottom_right)
        self._polygon = Polygon([top_left, top_right, bottom_right, bottom_left])

        top_width = self._top_left.distance(self._top_right)
        bottom_width = self._bottom_left.distance(self._bottom_right)
        left_height = self._top_left.distance(self._bottom_left)
        right_height = self._top_right.distance(self._bottom_right)
        max_width = max(int(top_width), int(bottom_width))
        max_height = max(int(left_height), int(right_height))
        self._width = max_width
        self._height = max_height

        frame_coordinates = np.float32(
            [
                top_left,
                top_right,
                bottom_left,
                bottom_right,
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
        self.transform_matrix = cv.getPerspectiveTransform(frame_coordinates, screen_coordinates)

    @property
    def top_left(self):
        return self._top_left.coords[0]

    @property
    def top_right(self):
        return self._top_right.coords[0]

    @property
    def bottom_left(self):
        return self._bottom_left.coords[0]

    @property
    def bottom_right(self):
        return self._bottom_right.coords[0]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def intersection_area(self, other):
        if isinstance(other, ConvexQuadrangleABC):
            if isinstance(other, ConvexQuadrangle):
                other_polygon = other._polygon
            else:
                other_polygon = Polygon([
                    other.top_left,
                    other.top_right,
                    other.bottom_right,
                    other.bottom_left,
                ])
            intersection = self._polygon.intersection(other_polygon)
            return intersection.area
        return NotImplemented

    def transform(self, frame_image):
        return cv.warpPerspective(
            frame_image,
            self.transform_matrix,
            (self.width, self.height),
            borderMode=cv.BORDER_CONSTANT,
            borderValue=COLOR_TRANSPARENT,
        )
