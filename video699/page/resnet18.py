# -*- coding: utf-8 -*-

r"""This module implements a page detector that matches last hidden ResNet18 layer activations for
document page image data with last hidden ResNet18 layer activations for projection screen image data.
Related classes and functions are also implemented.

"""

from itertools import chain
from logging import getLogger

import cv2 as cv
from annoy import AnnoyIndex

import torch
import torch.utils.data
from torch.autograd import Variable
from torchvision import models, transforms

from video699.configuration import get_configuration
from video699.interface import PageDetectorABC

LOGGER = getLogger(__name__)
CONFIGURATION = get_configuration()['PytorchResnet18PageDetector']
RESNET18_INPUT_SIZE = 224
RESNET18_OUTPUT_SIZE = 25088
RESNET18_MODEL = models.vgg16(pretrained=True).cuda()
RESNET18_MODEL.eval()
FEATURE_EXTRACTOR = torch.nn.Sequential(*list(RESNET18_MODEL.children())[:-2])


class Resnet18Dataset(torch.utils.data.Dataset):
    def __init__(self, images):
        super(Resnet18Dataset, self).__init__()
        self.images = list(images)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        self.transforms = transforms.Compose([
            to_tensor,
            normalize,
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        img = cv.cvtColor(img.render(RESNET18_INPUT_SIZE, RESNET18_INPUT_SIZE), cv.COLOR_BGRA2RGB)
        return self.transforms(img)


def _last_hidden_resnet18_layer(images):
    r"""

    Parameters
    ----------
    images : iterable of ImageABC
        Images.

    Returns
    -------
    activations : iterable of array_like
        The last hidden ResNet18 layer activations for the images.
    """
    batch_size = CONFIGURATION.getint('batch_size')
    dataset = Resnet18Dataset(images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    for image_batch in loader:
        image_batch = Variable(image_batch.cuda())
        activation_batch = FEATURE_EXTRACTOR(image_batch)
        for activations in activation_batch:
            yield activations.squeeze().flatten()


class PyTorchResNet18PageDetector(PageDetectorABC):
    r"""A page detector using approximate nearest neighbor search of last ResNet18 layer activations.

    Parameters
    ----------
    documents : set of DocumentABC
        The provided document pages.
    """

    def __init__(self, documents):
        annoy_n_trees = CONFIGURATION.getint('annoy_n_trees')
        annoy_distance_metric = CONFIGURATION['distance_metric']
        LOGGER.debug('Building an ANNOY index with {} trees'.format(annoy_n_trees))
        annoy_index = AnnoyIndex(RESNET18_OUTPUT_SIZE, metric=annoy_distance_metric)
        pages = dict()
        for page_index, (page, page_activations) in enumerate(
                zip(
                    chain(*documents),
                    _last_hidden_resnet18_layer(chain(*documents)),
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
        for screen, screen_activations in zip(screens, _last_hidden_resnet18_layer(screens)):
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
