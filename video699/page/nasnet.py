# -*- coding: utf-8 -*-

r"""This module implements a screen event detector that matches last hidden NASNet layer activations
for document page image data with last hidden NASNet layer activations for projection screen image
data. Related classes and functions are also implemented.

"""


from itertools import chain
from logging import getLogger

from annoy import AnnoyIndex
import cv2 as cv
from keras.applications.nasnet import NASNetLarge, preprocess_input
import numpy as np

from ..common import get_batches
from ..configuration import get_configuration
from ..interface import PageDetectorABC


LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['NASNetPageDetector']
NASNET_INPUT_SIZE = 331
NASNET_OUTPUT_SIZE = 4032
NASNET_MODEL = NASNetLarge(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(NASNET_INPUT_SIZE, NASNET_INPUT_SIZE, 3),
    pooling='max',
)


def _last_hidden_vgg16_layer(images):
    r"""Produces the last hidden NASNet layer activations for images.

    Parameters
    ----------
    images : iterable of ImageABC
        Images.

    Returns
    -------
    activations : iterable of array_like
        The last hidden NASNet layer activations for the images.
    """

    batch_size = CONFIGURATION.getint('batch_size')

    for image_batch in get_batches(images, batch_size):
        image_batch_rgb = preprocess_input(
            np.array([
                cv.cvtColor(
                    image.render(NASNET_INPUT_SIZE, NASNET_INPUT_SIZE),
                    cv.COLOR_BGRA2RGB,
                )
                for image in image_batch
            ], dtype=np.float32)
        )
        activation_batch = NASNET_MODEL.predict([image_batch_rgb]).reshape(-1, NASNET_OUTPUT_SIZE)
        for activations in activation_batch:
            yield activations


class KerasNASNetPageDetector(PageDetectorABC):
    r"""A page detector using approximate nearest neighbor search of last NASNet layer activations.

    The NASNet model is based on the paper by Zoph et al. [Zoph18]_.

    .. [Zoph18]. Zoph, Barret and Vasudevan, Vijay and Shlens, Jonathon and Le, Quoc V. "Learning
    Transferable Architectures for Scalable Image Recognition." *arXiv*. 2018.
    `URL <https://arxiv.org/abs/1707.07012>`_

    Parameters
    ----------
    documents : set of DocumentABC
        The provided document pages.
    """

    def __init__(self, documents):
        annoy_n_trees = CONFIGURATION.getint('annoy_n_trees')
        annoy_distance_metric = CONFIGURATION['distance_metric']
        LOGGER.debug('Building an ANNOY index with {} trees'.format(annoy_n_trees))
        annoy_index = AnnoyIndex(NASNET_OUTPUT_SIZE, metric=annoy_distance_metric)
        pages = dict()
        for page_index, (page, page_activations) in enumerate(
                    zip(
                        chain(*documents),
                        _last_hidden_vgg16_layer(chain(*documents)),
                    )
                ):
            annoy_index.add_item(page_index, page_activations)
            pages[page_index] = page
        annoy_index.build(annoy_n_trees)

        self._annoy_index = annoy_index
        self._pages = pages

    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        annoy_search_k = CONFIGURATION.getint('annoy_search_k')
        num_nearest_pages = CONFIGURATION.getint('num_nearest_pages')
        # max_distance = CONFIGURATION.getfloat('max_distance')
        max_distance = float('inf')

        annoy_index = self._annoy_index
        pages = self._pages

        detected_pages = {}
        screens = set(screen for screen, _ in chain(appeared_screens, existing_screens))
        for screen, screen_activations in zip(screens, _last_hidden_vgg16_layer(screens)):
            page_indices, page_distances = annoy_index.get_nns_by_vector(
                screen_activations,
                num_nearest_pages,
                search_k=annoy_search_k,
                include_distances=True,
            )
            closest_matching_page = None
            for page_index, page_distance in zip(page_indices, page_distances):
                if page_distance < max_distance:
                    closest_matching_page = pages[page_index]
                    break
            detected_pages[screen] = closest_matching_page

        return detected_pages
