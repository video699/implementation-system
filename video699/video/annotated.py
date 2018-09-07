# -*- coding: utf-8 -*-

"""This module implements reading a sample of a video from a dataset with XML human annotations, and
related classes.

"""

import json
from logging import getLogger
import os
import re

import cv2 as cv
from dateutil.parser import parse as datetime_parse
from lxml import etree
import numpy as np

from ..convex_quadrangle import ConvexQuadrangle
from ..document.image import ImagePage
from ..frame.image import ImageFrame
from ..interface import VideoABC, FrameABC, DocumentABC, PageABC


LOGGER = getLogger(__name__)
RESOURCES_PATHNAME = os.path.join(os.path.dirname(__file__), 'annotated')
DATASET_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'dataset.xml')
DOCUMENT_ANNOTATIONS = None
FRAME_ANNOTATIONS = None
# VIDEO_ANNOTATIONS = None
VIDEOS = None
URI_REGEX = re.compile(
    r'https?://is\.muni\.cz/auth/el/(?P<faculty>\d+)/(?P<term>[^/]+)/(?P<course>[^/]+)/um/vi/'
    r'\?videomuni=(?P<filename>[^-]+-(?P<room>[^-]+)'
    r'-(?P<date>(?P<year>\d{4})(?P<month>\d{2})(?P<day_of_month>\d{2}))\.\w+)'
)


def _init_dataset():
    """Reads human annotations from an XML dataset, converts them into objects and sorts them.

    """
    global DOCUMENT_ANNOTATIONS
    global FRAME_ANNOTATIONS
    global VIDEO_ANNOTATIONS
    global VIDEOS
    LOGGER.debug('Loading dataset {}'.format(DATASET_PATHNAME))
    videos = etree.parse(DATASET_PATHNAME)
    videos.xinclude()
    DOCUMENT_ANNOTATIONS = {
        video.attrib['uri']: {
            document.attrib['filename']: _DocumentAnnotations(
                filename=document.attrib['filename'],
                pages={
                    page.attrib['key']: _PageAnnotations(
                        key=page.attrib['key'],
                        number=int(page.attrib['number']),
                        filename=page.attrib['filename'],
                        vgg256=VGG256Features(*json.loads(page.attrib['vgg256'])),
                    ) for page in document.findall('./page')
                },
            ) for document in video.findall('./documents/document')
        } for video in videos.findall('./video')
    }
    FRAME_ANNOTATIONS = {
        video.attrib['uri']: {
            int(frame.attrib['number']): _FrameAnnotations(
                filename=frame.attrib['filename'],
                number=int(frame.attrib['number']),
                vgg256=VGG256Features(*json.loads(frame.attrib['vgg256'])),
            ) for frame in video.findall('./frames/frame')
        } for video in videos.findall('./video')
    }
    VIDEO_ANNOTATIONS = {
        video.attrib['uri']: {
            'uri': video.attrib['uri'],
            'dirname': video.attrib['dirname'],
            'fps': int(video.attrib['fps']),
            'width': int(video.attrib['width']),
            'height': int(video.attrib['height']),
            'frames': {
                int(frame.attrib['number']): {
                    'vgg256': VGG256Features(*json.loads(frame.attrib['vgg256'])),
                    'filename': frame.attrib['filename'],
                    'screens': [
                        {
                            'vgg256': VGG256Features(*json.loads(screen.attrib['vgg256'])),
                            'coordinates': ConvexQuadrangle(
                                top_left=(
                                    int(screen.attrib['x0']),
                                    int(screen.attrib['y0']),
                                ),
                                top_right=(
                                    int(screen.attrib['x1']),
                                    int(screen.attrib['y1']),
                                ),
                                bottom_left=(
                                    int(screen.attrib['x2']),
                                    int(screen.attrib['y2']),
                                ),
                                bottom_right=(
                                    int(screen.attrib['x3']),
                                    int(screen.attrib['y3']),
                                ),
                            ),
                            'condition': screen.attrib['condition'],
                            'keyrefs': {
                                keyref.text: keyref.attrib['similarity']
                                for keyref in screen.findall('./keyrefs/keyref')
                            },
                        } for screen in frame.findall('./screens/screen')
                    ]
                } for frame in video.findall('./frames/frame')
            }
        } for video in videos.findall('./video')
    }
    VIDEOS = set(
        AnnotatedSampledVideo(video_annotations["uri"])
        for video_annotations in VIDEO_ANNOTATIONS.values()
    )


def get_videos():
    """Returns all videos from a XML dataset.

    Returns
    -------
    videos : iterator of AnnotatedSampledVideo
        All videos from a XML dataset.
    """
    return iter(VIDEOS)


class VGG256Features(object):
    """Two feature vectors obtained from the 256-dimensional last hidden layers of [VGG]_ ConvNets.

    .. _Imagenet: http://image-net.org
    .. _Places2: http://places2.csail.mit.edu/
    .. [VGG] Simonyan, Karen & Zisserman, Andrew. (2014). Very Deep Convolutional Networks for
       Large-Scale Image Recognition. `arXiv 1409.1556 <https://arxiv.org/abs/1409.1556>`_.

    Parameters
    ----------
    imagenet : array_like
        A 256-dimensional feature vector obtained from a network trained on the Imagenet_ dataset.
    imagenet_and_places2 : array_like
        A 256-dimensional feature vector obtained from a network trained on the Imagenet_, and
        Places2_ datasets.

    Attributes
    ----------
    imagenet : np.array
        A 256-dimensional feature vector obtained from a network trained on the Imagenet_ dataset.
        The feature vector is stored in a NumPy array of 64-bit floats.
    imagenet_and_places2 : np.array
        A 256-dimensional feature vector obtained from a network trained on the Imagenet_, and
        Places2_ datasets. The feature vector is stored in a NumPy array of 64-bit floats.
    """
    def __init__(self, imagenet, imagenet_and_places2):
        self.imagenet = np.array(imagenet, dtype=float)
        self.imagenet_and_places2 = np.array(imagenet_and_places2, dtype=float)


class _DocumentAnnotations(object):
    """Human annotations associated with a single document.

    Parameters
    ----------
    filename : str
        The filename of the corresponding PDF document. The filename is unique in the video.
    pages : dict of (str, _PageAnnotations)
        A map between page keys, and human-annotations associated with the pages of the document.

    Attributes
    ----------
    filename : str
        The filename of the corresponding PDF document. The filename is unique in the video.
    pages : dict of (str, _PageAnnotations)
        A map between page keys, and human-annotations associated with the pages of the document.
    """

    def __init__(self, filename, pages):
        self.filename = filename
        self.pages = pages


class _PageAnnotations(object):
    """Human annotations associated with a single page of a document.

    Parameters
    ----------
    key : str
        An identifier of a page in a document. The identifier is unique in the video associated with
        the document.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
    filename : str
        The filename of the corresponding document page image. The filename is unique in the video
        associated with the document.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the page image into VGG ConvNets.

    Attributes
    ----------
    key : str
        An identifier of a page in a document. The identifier is unique in the video associated with
        the document.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
    filename : str
        The filename of the corresponding document page image. The filename is unique in the video
        associated with the document.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the page image into VGG ConvNets.
    """

    def __init__(self, key, number, filename, vgg256):
        self.key = key
        self.number = number
        self.filename = filename
        self.vgg256 = vgg256


class AnnotatedSampledVideoDocumentPage(PageABC):
    """A single page of a document extracted from a dataset with XML human annotations.

    Parameters
    ----------
    document : AnnotatedSampledVideoDocument
        The document containing the page.
    key : str
        A page identifier. The identifier is unique in the video associated with the document.

    Attributes
    ----------
    document : DocumentABC
        The document containing the page.
    number : int
        The page number, i.e. the position of the page in the document. Page indexing is one-based,
        i.e. the first page has number 1.
    filename : str
        The filename of the corresponding document page image. The filename is unique in the video
        associated with the document.
    pathname : str
        The full pathname of the corresponding document page image. The pathname is unique in the
        video associated with the document.
    key : str
        A page identifier. The identifier is unique in the video associated with the document.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the page image into VGG ConvNets.
    """

    def __init__(self, document, key):
        self._document = document
        self.key = key

        page_annotations = DOCUMENT_ANNOTATIONS[document.video.uri][document.filename].pages[key]
        number = page_annotations.number
        self.filename = page_annotations.filename
        self.vgg256 = page_annotations.vgg256

        page_image = cv.imread(self.pathname)
        self._page = ImagePage(document, number, page_image)

    @property
    def document(self):
        return self._document

    @property
    def number(self):
        return self._page.number

    @property
    def pathname(self):
        pathname = os.path.join(
            self.document.video.dirname,
            self.filename,
        )
        return pathname

    def image(self, width, height):
        return self._page.image(width, height)


class AnnotatedSampledVideoDocument(DocumentABC):
    """A sequence of images forming a document extracted from a dataset with XML human annotations.

    Parameters
    ----------
    video : AnnotatedSampledVideo
        The video associated with this document.
    filename : str
        The filename of the corresponding PDF document. The filename is unique in the video.

    Attributes
    ----------
    video : AnnotatedSampledVideo
        The video associated with this document.
    filename : str
        The filename of the corresponding PDF document. The filename is unique in the video.
    pathname : str
        The full pathname of the corresponding PDF document. The pathname is unique in the video.
    title : str or None
        The title of a document.
    author : str or None
        The author of a document.
    """

    def __init__(self, video, filename):
        self.video = video
        self.filename = filename

        document_annotations = DOCUMENT_ANNOTATIONS[video.uri][filename]
        self._pages = set(
            AnnotatedSampledVideoDocumentPage(
                self,
                page_annotations.key,
            ) for page_annotations in document_annotations.pages.values()
        )

    @property
    def title(self):
        return None

    @property
    def author(self):
        return None

    @property
    def pathname(self):
        pathname = os.path.join(
            self.video.dirname,
            self.filename,
        )
        return pathname

    def __iter__(self):
        return iter(self._pages)


class _FrameAnnotations(object):
    """Human annotations associated with a single frame of a video.

    Parameters
    ----------
    filename : str
        The filename of the corresponding video frame image. The filename is unique in the video.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1. The frame number is unique in the video.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the frame image into VGG ConvNets.

    Attributes
    ----------
    filename : str
        The filename of the corresponding video frame image. The filename is unique in the video.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1. The frame number is unique in the video.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the frame image into VGG ConvNets.
    """

    def __init__(self, filename, number, vgg256):
        self.filename = filename
        self.number = number
        self.vgg256 = vgg256


class AnnotatedSampledVideoFrame(FrameABC):
    """A frame of a video extracted from a dataset with XML human annotations.

    Parameters
    ----------
    video : VideoABC
        The video containing the frame.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1. The frame number is unique in the video.

    Attributes
    ----------
    video : VideoABC
        The video containing the frame.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1. The frame number is unique in the video.
    filename : str
        The filename of the corresponding video frame image. The filename is unique in the video.
    pathname : str
        The full pathname of the corresponding video frame image. The pathname is unique in the
        video.
    image : array_like
        The image data of the frame represented as an OpenCV CV_8UC3 BGR matrix.
    width : int
        The width of the image data.
    height : int
        The height of the image data.
    datetime : aware datetime
        The date, and time at which the frame was captured.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the frame image into VGG ConvNets.
    """

    def __init__(self, video, number):
        self._video = video

        frame_annotations = FRAME_ANNOTATIONS[video.uri][number]
        self.filename = frame_annotations.filename
        self.vgg256 = frame_annotations.vgg256

        frame_image = cv.imread(self.pathname)
        self._frame = ImageFrame(video, number, frame_image)

    @property
    def video(self):
        return self._video

    @property
    def number(self):
        return self._frame.number

    @property
    def pathname(self):
        pathname = os.path.join(
            self.video.dirname,
            self.filename,
        )
        return pathname

    @property
    def image(self):
        return self._frame.image


class AnnotatedSampledVideo(VideoABC):
    """A sample of a video file extracted from a dataset with XML human annotations.

    Parameters
    ----------
    uri : str
        The URI of the video file. The URI is unique in the dataset.

    Attributes
    ----------
    uri : str
        The URI of the video file. The URI is unique in the dataset.
    dirname : str
        The pathname of the directory, where the frames, documents, and XML human annotations
        associated with the video are stored.
    filename : str
        The filename of the video file.
    fps : scalar
        The framerate of the video in frames per second.
    width : int
        The width of the video.
    height : int
        The height of the video.
    datetime : aware datetime
        The date, and time at which the video was captured.
    documents : set of PDFDocument
        The documents associated with the video.
    """

    def __init__(self, uri):
        self.uri = uri
        match = re.fullmatch(URI_REGEX, uri)
        self.filename = match.group('filename')
        self._datetime = datetime_parse(
            '{0}-{1}-{2}T00:00:00+00:00'.format(
                match.group('year'),
                match.group('month'),
                match.group('day_of_month'),
            ),
        )

        video_annotations = VIDEO_ANNOTATIONS[uri]
        self.dirname = video_annotations['dirname']
        self._fps = video_annotations['fps']
        self._width = video_annotations['width']
        self._height = video_annotations['height']

        self._frames = sorted([
            AnnotatedSampledVideoFrame(
                self,
                frame_annotations.number
            ) for frame_annotations in FRAME_ANNOTATIONS[uri].values()
        ])

        self.documents = set(
            AnnotatedSampledVideoDocument(
                self,
                document_annotations.filename,
            ) for document_annotations in DOCUMENT_ANNOTATIONS[uri].values()
        )

    @property
    def fps(self):
        return self._fps

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def datetime(self):
        return self._datetime

    def __iter__(self):
        return iter(self._frames)


_init_dataset()
