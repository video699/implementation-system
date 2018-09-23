# -*- coding: utf-8 -*-

"""This module implements a convex quadranfle index that uses the R-tree data structure to
efficiently retrieve convex quadrangles. The implementation uses the libspatialindex library exposed
through the Rtree Python package.

"""

import rtree

from ..interface import ConvexQuadrangleIndexABC


class RTreeConvexQuadrangleIndex(ConvexQuadrangleIndexABC):
    """A convex quadrangle index that uses the R-tree structure to efficiently retrieve quadrangles.

    Parameters
    ----------
    quadrangles : iterable of ConvexQuadrangleABC
        The initial convex quadrangles in the index.

    Attributes
    ----------
    quadrangles : read-only set-like object of ConvexQuadrangleABC
        The convex quadrangles in the index.
    """

    def __init__(self, quadrangles=()):
        self._quadrangles = {id(quadrangle): quadrangle for quadrangle in quadrangles}
        self._quadrangle_ids = {
            quadrangle: quadrangle_id
            for quadrangle_id, quadrangle in self._quadrangles.items()
        }
        self._index = rtree.index.Index([
            (quadrangle_id, (*quadrangle.top_left_bound, *quadrangle.bottom_right_bound), None)
            for quadrangle_id, quadrangle in self._quadrangles.items()
        ])

    @property
    def quadrangles(self):
        return self._quadrangles.values()

    def add(self, quadrangle):
        if quadrangle not in self.quadrangles:
            self._quadrangles[id(quadrangle)] = quadrangle
            self._quadrangle_ids[quadrangle] = id(quadrangle)
            quadrangle_id = id(quadrangle)
            coordinates = (*quadrangle.top_left_bound, *quadrangle.bottom_right_bound)
            self._index.insert(quadrangle_id, coordinates)

    def discard(self, quadrangle):
        if quadrangle in self.quadrangles:
            quadrangle_id = self._quadrangle_ids[quadrangle]
            coordinates = (*quadrangle.top_left_bound, *quadrangle.bottom_right_bound)
            del self._quadrangles[quadrangle_id]
            del self._quadrangle_ids[quadrangle]
            self._index.delete(quadrangle_id, coordinates)

    def clear(self):
        self._quadrangles.clear()
        self._quadrangle_ids.clear()
        self._index = rtree.index.Index()

    def intersection_areas(self, input_quadrangle):
        coordinates = (*input_quadrangle.top_left_bound, *input_quadrangle.bottom_right_bound)
        intersection = self._index.intersection(coordinates)
        intersection_areas = {}
        for indexed_quadrangle_id in intersection:
            indexed_quadrangle = self._quadrangles[indexed_quadrangle_id]
            intersection_area = input_quadrangle.intersection_area(indexed_quadrangle)
            if intersection_area > 0:
                intersection_areas[indexed_quadrangle] = intersection_area
        return intersection_areas
