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
    current_quadrangle : ConvexQuadrangleABC
        The latest coordinates of the moving convex quadrangle.
    window_size : int or None, optional
        The maximum number of previous time frames for which the quadrangle movements are stored. If
        ``None`` or unspecified, then the number of time frames is unbounded.

    Attributes
    ----------
    quadrangle_id : str
        An identifier unique among the :class:`TrackedConvexQuadrangleABC` tracked identifiers
        produced by a convex quadrangle tracker.
    current_quadrangle : ConvexQuadrangleABC
        The latest coordinates of the moving convex quadrangle.

    Raises
    ------
    ValueError
        If the window size is less than one.
    """

    def __init__(self, quadrangle_id, current_quadrangle, window_size=None):
        self._quadrangle_id = quadrangle_id
        if window_size is not None and window_size < 1:
            raise ValueError('The window size must not be less than one')
        self._quadrangles = deque((current_quadrangle,), maxlen=window_size)

    @property
    def quadrangle_id(self):
        return self._quadrangle_id

    def add(self, quadrangle):
        self._quadrangles.append(quadrangle)

    def __iter__(self):
        return iter(self._quadrangles)

    def __reversed__(self):
        return reversed(self._quadrangles)
