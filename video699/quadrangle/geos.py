# -*- coding: utf-8 -*-

"""This module implements a coordinate map between a video frame and projection screen coordinate
systems based on a quadrangle specifying the screen corners in the video frame coordinate system.
The implementation uses the GEOS library exposed through the Shapely Python package.

"""

import cv2 as cv
import numpy as np
from shapely.geometry import Point, Polygon

from ..common import change_aspect_ratio_by_upscaling, COLOR_RGBA_TRANSPARENT
from ..configuration import get_configuration
from ..interface import ConvexQuadrangleABC

CONFIGURATION = get_configuration()['GEOSConvexQuadrangle']
RESCALE_INTERPOLATION = cv.__dict__[CONFIGURATION['rescale_interpolation']]


class GEOSConvexQuadrangle(ConvexQuadrangleABC):
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
    aspect_ratio : Fraction or None, optional
        The aspect ratio of the quadrangle in a screen coordinate system. If ``None`` or
        unspecified, the aspect ratio will be the ratio between the longest adjacent sides of the
        quadrangle in the video frame coordinate system.

    Attributes
    ----------
    top_left : (scalar, scalar)
        The top left corner of the quadrangle in a video frame coordinate system.
    top_right : (scalar, scalar)
        The top right corner of the quadrangle in a video frame coordinate system.
    top_right_bound : (scalar, scalar)
        The top right corner of the minimal bounding box that bounds the quadrangle in a video frame
        coordinate system.
    bottom_left : (scalar, scalar)
        The bottom left corner of the quadrangle in a video frame coordinate system.
    bottom_left_bound : (scalar, scalar)
        The bottom left corner of the minimal bounding box that bounds the quadrangle in a video
        frame coordinate system.
    bottom_right : (scalar, scalar)
        The bottom right corner of the quadrangle in a video frame coordinate system.
    width : int
        The width of the quadrangle in a screen coordinate system.
    height : int
        The height of the quadrangle in a screen coordinate system.
    area : scalar
        The area of the screen in the video frame coordinate system.
    """

    def __init__(self, top_left, top_right, bottom_left, bottom_right, aspect_ratio=None):
        self._top_left = Point(top_left)
        self._top_right = Point(top_right)
        self._bottom_left = Point(bottom_left)
        self._bottom_right = Point(bottom_right)
        self._hash = hash((self.top_left, self.top_right, self.bottom_left, self.bottom_right))
        self._polygon = Polygon([top_left, top_right, bottom_right, bottom_left])

        top_width = self._top_left.distance(self._top_right)
        bottom_width = self._bottom_left.distance(self._bottom_right)
        left_height = self._top_left.distance(self._bottom_left)
        right_height = self._top_right.distance(self._bottom_right)
        max_width = max(int(top_width), int(bottom_width))
        max_height = max(int(left_height), int(right_height))

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

        if aspect_ratio is None:
            self._width = max_width
            self._height = max_height
        else:
            self._width, self._height = change_aspect_ratio_by_upscaling(
                max_width,
                max_height,
                aspect_ratio,
            )
            stretch_x = self.width / max_width
            stretch_y = self.height / max_height
            self.transform_matrix = np.array([
                (stretch_x, 0, 0),
                (0, stretch_y, 0),
                (0, 0, 1),
            ]).dot(self.transform_matrix)

    @property
    def top_left(self):
        return self._top_left.coords[0]

    @property
    def top_right(self):
        return self._top_right.coords[0]

    @property
    def top_left_bound(self):
        return self._polygon.bounds[0:2]

    @property
    def bottom_left(self):
        return self._bottom_left.coords[0]

    @property
    def bottom_right(self):
        return self._bottom_right.coords[0]

    @property
    def bottom_right_bound(self):
        return self._polygon.bounds[2:4]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def area(self):
        return self._polygon.area

    def intersection_area(self, other):
        if isinstance(other, ConvexQuadrangleABC):
            if isinstance(other, GEOSConvexQuadrangle):
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
            borderValue=COLOR_RGBA_TRANSPARENT,
            flags=RESCALE_INTERPOLATION,
        )

    def __hash__(self):
        return self._hash
