# -*- coding: utf-8 -*-

"""This module implements a moving convex quadrangle using the deque data structure.

"""

from collections import deque

from ..interface import MovingConvexQuadrangleABC


class DequeMovingConvexQuadrangle(MovingConvexQuadrangleABC):
    """A convex quadrangle that moves in time represented by a deque.

    Parameters
    ----------
    quadrangle_id : str
        An identifier unique among the :class:`TrackedConvexQuadrangleABC` tracked identifiers
        produced by a convex quadrangle tracker.
    window_size : int or None, optional
        The maximum number of previous time frames for which the quadrangle movements are stored. If
        ``None`` or unspecified, then the number of time frames is unbounded.

    Attributes
    ----------
    quadrangle_id : str
        An identifier unique among the :class:`TrackedConvexQuadrangleABC` tracked identifiers
        produced by a convex quadrangle tracker.
    """

    def __init__(self, quadrangle_id, window_size=None):
        self._quadrangle_id = quadrangle_id
        self._quadrangles = deque(maxlen=window_size)

    @property
    def quadrangle_id(self):
        return self._quadrangle_id

    def add(self, quadrangle):
        self._quadrangles.append(quadrangle)

    def __iter__(self):
        return iter(self._quadrangles)

    def __reversed__(self):
        return reversed(self._quadrangles)
