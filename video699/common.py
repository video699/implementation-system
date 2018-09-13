# -*- coding: utf-8 -*-

"""This module implements commonly useful constants, and functions.

"""

from math import floor, ceil


COLOR_RGBA_TRANSPARENT = (0, 0, 0, 0)


def rescale_and_keep_aspect_ratio(original_width, original_height, new_width, new_height):
    """Returns new dimensions, and margins that rescale an image and keep its aspect ratio.

    Notes
    -----
    The rescaled image is vertically and horizontally centered and its dimensions do not exceed
    the new dimensions of the image either vertically or horizontally.

    Parameters
    ----------
    original_width : int
        The original width of an image.
    original_height : int
        The original height of an image.
    new_width : int
        The new width of the image including the margins.
    new_height : int
        The new height of the image including the margins.

    Returns
    -------
    rescaled_width : int
        The width of the image after rescaling.
    rescaled_height : int
        The height of the image after rescaling.
    top_margin : int
        The width of the top margin of the rescaled image.
    bottom_margin : int
        The width of the bottom margin of the rescaled image.
    left_margin : int
        The width of the left margin of the rescaled image.
    right_margin : int
        The width of the right margin of the rescaled image.
    """
    aspect_ratio = min(new_width / original_width, new_height / original_height)
    rescaled_width = int(round(original_width * aspect_ratio))
    rescaled_height = int(round(original_height * aspect_ratio))
    margin_width = (new_width - rescaled_width) / 2
    margin_height = (new_height - rescaled_height) / 2
    top_margin = floor(margin_height)
    bottom_margin = ceil(margin_height)
    left_margin = floor(margin_width)
    right_margin = ceil(margin_width)
    return (
        rescaled_width,
        rescaled_height,
        top_margin,
        bottom_margin,
        left_margin,
        right_margin,
    )
