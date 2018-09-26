# -*- coding: utf-8 -*-

"""This module implements a convex quadrangle index that uses the R-tree data structure to
efficiently retrieve convex quadrangles. The implementation uses the libspatialindex library exposed
through the Rtree Python package. A convex quadrangle tracker that uses the convex quadrangle index
is also implemented.

"""

import rtree

from ..interface import ConvexQuadrangleIndexABC, ConvexQuadrangleTrackerABC
from .deque import DequeMovingConvexQuadrangle


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


class RTreeConvexQuadrangleTracker(ConvexQuadrangleTrackerABC):
    """A convex quadrangle tracker that uses the :class:`RTreeConvexQuadrangleIndex` index.

    Parameters
    ----------
    window_size : int or None, optional
        The maximum number of previous time frames for which the quadrangle movements are stored. If
        ``None`` or unspecified, then the number of time frames is unbounded.
    """

    def __init__(self, window_size=None):
        self._window_size = window_size
        self._moving_quadrangles = {}
        self._quadrangle_index = RTreeConvexQuadrangleIndex()

    def update(self, current_quadrangles):
        """Records convex quadrangles that exist in the current time frame.

        The convex quadrangles in the *current* time frame in the order of iteration are compared
        with the convex quadrangles in the *previous* time frame. The current quadrangles that
        intersect no previous quadrangles are added to the tracker. The current quadrangles that
        intersect at least one previous quadrangle are considered to be the current position of the
        previous quadrangle with the largest intersection area. The previous quadrangles that cross
        no current quadrangles are removed from the tracker.

        Parameters
        ----------
        current_quadrangles : iterable of ConvexQuadrangleABC
            The convex quadrangles in the current time frame.

        Returns
        -------
        appeared_quadrangles : set of MovingConvexQuadrangleABC
            The current quadrangles that intersect no previous quadrangles.
        existing_quadrangles : set of MovingConvexQuadrangleABC
            The current quadrangles that intersect at least one previous quadrangle.
        disappeared_quadrangles : set of MovingConvexQuadrangleABC
            The previous quadrangles that cross no current quadrangles.
        """

        appeared_quadrangles = set()
        existing_quadrangles = set()
        disappeared_quadrangles = set()
        window_size = self._window_size
        moving_quadrangles = self._moving_quadrangles
        quadrangle_index = self._quadrangle_index

        for quadrangle in current_quadrangles:
            intersection_areas = quadrangle_index.intersection_areas(quadrangle)
            if intersection_areas:
                (previous_quadrangle, _), *__ = sorted(
                    intersection_areas.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                moving_quadrangle = moving_quadrangles[previous_quadrangle]
                moving_quadrangle.add(quadrangle)
                quadrangle_index.remove(previous_quadrangle)
                del moving_quadrangles[previous_quadrangle]
                existing_quadrangles.add(moving_quadrangle)
            else:
                moving_quadrangle = DequeMovingConvexQuadrangle(
                    quadrangle,
                    window_size,
                )
                appeared_quadrangles.add(moving_quadrangle)

        current_moving_quadrangles = appeared_quadrangles | existing_quadrangles
        for moving_quadrangle in current_moving_quadrangles:
            quadrangle = next(reversed(moving_quadrangle))
            moving_quadrangles[quadrangle] = moving_quadrangle
            quadrangle_index.add(quadrangle)

        for previous_quadrangle, moving_quadrangle in list(moving_quadrangles.items()):
            if moving_quadrangle not in current_moving_quadrangles:
                quadrangle_index.remove(previous_quadrangle)
                del moving_quadrangles[previous_quadrangle]
                disappeared_quadrangles.add(moving_quadrangle)

        return (appeared_quadrangles, existing_quadrangles, disappeared_quadrangles)

    def __iter__(self):
        return iter(self._moving_quadrangles)

    def __len__(self):
        return len(self._moving_quadrangles)
