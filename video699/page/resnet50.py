# -*- coding: utf-8 -*-

r"""This module implements a page detector that matches last hidden ResNet50 layer activations for
document page image data with last hidden ResNet50 layer activations for projection screen image data.
Related classes and functions are also implemented.

"""


from itertools import chain
from logging import getLogger

from annoy import AnnoyIndex
import cv2 as cv
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np

from ..common import get_batches
from ..configuration import get_configuration
from ..interface import PageDetectorABC


LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['KerasResnet50PageDetector']
RESNET50_INPUT_SIZE = 224
RESNET50_OUTPUT_SIZE = 100352
RESNET50_MODEL = ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(RESNET50_INPUT_SIZE, RESNET50_INPUT_SIZE, 3),
    pooling=None,
)


def _last_hidden_resnet50_layer(images):
    r"""Produces the last hidden ResNet50 layer activations for images.

    Parameters
    ----------
    images : iterable of ImageABC
        Images.

    Returns
    -------
    activations : iterable of array_like
        The last hidden ResNet50 layer activations for the images.
    """

    batch_size = CONFIGURATION.getint('batch_size')

    for image_batch in get_batches(images, batch_size):
        image_batch_rgb = preprocess_input(
            np.array([
                cv.cvtColor(
                    image.render(RESNET50_INPUT_SIZE, RESNET50_INPUT_SIZE),
                    cv.COLOR_BGRA2RGB,
                )
                for image in image_batch
            ], dtype=np.float32)
        )
        activation_batch = RESNET50_MODEL.predict([image_batch_rgb]).reshape(-1, RESNET50_OUTPUT_SIZE)
        for activations in activation_batch:
            yield activations


class KerasResNet50PageDetector(PageDetectorABC):
    r"""A page detector using approximate nearest neighbor search of last ResNet50 layer activations.

    The ResNet50 model is based on the paper by Simonyan and Zisserman [Simoyan15]_.

    .. [Simoyan15]. Simoyan, Karen and Zisserman, Andrew. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." *arXiv*. 2015. `URL <https://arxiv.org/abs/1409.1556>`_

    Parameters
    ----------
    documents : set of DocumentABC
        The provided document pages.
    """

    def __init__(self, documents):
        annoy_n_trees = CONFIGURATION.getint('annoy_n_trees')
        annoy_distance_metric = CONFIGURATION['distance_metric']
        LOGGER.debug('Building an ANNOY index with {} trees'.format(annoy_n_trees))
        annoy_index = AnnoyIndex(RESNET50_OUTPUT_SIZE, metric=annoy_distance_metric)
        pages = dict()
        for page_index, (page, page_activations) in enumerate(
                    zip(
                        chain(*documents),
                        _last_hidden_resnet50_layer(chain(*documents)),
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
        max_distance = CONFIGURATION.getfloat('max_distance')

        annoy_index = self._annoy_index
        pages = self._pages

        detected_pages = {}
        screens = set(screen for screen, _ in chain(appeared_screens, existing_screens))
        for screen, screen_activations in zip(screens, _last_hidden_resnet50_layer(screens)):
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
