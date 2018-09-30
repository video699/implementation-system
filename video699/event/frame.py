# -*- coding: utf-8 -*-

"""This module defines interfaces, and abstract base classes for events that take place in a video
frame.

"""

from abc import abstractmethod
from functools import total_ordering

from ..interface import EventABC


@total_ordering
class FrameEventABC(EventABC):
    """An event that takes place in a video frame.

    Attributes
    ----------
    frame : FrameABC
        A frame in which the event takes place.
    """

    @property
    @abstractmethod
    def frame(self):
        pass

    def __hash__(self):
        return hash(self.frame)

    def __eq__(self, other):
        if isinstance(other, FrameEventABC):
            return self.frame == other.frame
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, FrameEventABC):
            return self.frame < other.frame
        return NotImplemented

    def __repr__(self):
        return '<{classname}, frame {frame}>'.format(
            classname=self.__class__.__name__,
            frame=self.frame,
        )
