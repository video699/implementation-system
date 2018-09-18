# -*- coding: utf-8 -*-

"""This module implements reading a sample of a video from a dataset with XML human annotations, and
related classes.

"""

import json
from logging import getLogger
import os
from pathlib import Path
import re

import cv2 as cv
from dateutil.parser import parse as datetime_parse
from lxml import etree
import numpy as np

from ..convex_quadrangle import ConvexQuadrangle
from ..document.image_file import ImageFileDocumentPage
from ..document.pdf import PDFDocument
from ..frame.image import ImageFrame
from ..interface import VideoABC, FrameABC, ScreenABC, ScreenDetectorABC, DocumentABC, PageABC


LOGGER = getLogger(__name__)
RESOURCES_PATHNAME = os.path.join(os.path.dirname(__file__), 'annotated')
DATASET_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'dataset.xml')
DOCUMENT_ANNOTATIONS = None
FRAME_ANNOTATIONS = None
VIDEO_ANNOTATIONS = None
VIDEOS = None
PAGES = None
SCREENS = None
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
    global PAGES
    global SCREENS
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
                screens=[
                    _ScreenAnnotations(
                        coordinates=ConvexQuadrangle(
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
                        condition=screen.attrib['condition'],
                        keyrefs={
                            keyref.text: _KeyRefAnnotations(
                                key=keyref.text,
                                similarity=keyref.attrib['similarity'],
                            ) for keyref in screen.findall('./keyrefs/keyref')
                        },
                        vgg256=VGG256Features(*json.loads(screen.attrib['vgg256'])),
                    ) for screen in frame.findall('./screens/screen')
                ],
                vgg256=VGG256Features(*json.loads(frame.attrib['vgg256'])),
            ) for frame in video.findall('./frames/frame')
        } for video in videos.findall('./video')
    }
    VIDEO_ANNOTATIONS = {
        video.attrib['uri']: _VideoAnnotations(
            uri=video.attrib['uri'],
            dirname=video.attrib['dirname'],
            num_frames=int(video.attrib['frames']),
            fps=int(video.attrib['fps']),
            width=int(video.attrib['width']),
            height=int(video.attrib['height']),
        ) for video in videos.findall('./video')
    }
    VIDEOS = {
        video_annotations.uri: AnnotatedSampledVideo(video_annotations.uri)
        for video_annotations in VIDEO_ANNOTATIONS.values()
    }
    PAGES = {
        video.uri: {
            page.key: page
            for document in video.documents.values()
            for page in document
        } for video in VIDEOS.values()
    }
    SCREENS = {
        video.uri: {
            frame.number: [
                AnnotatedSampledVideoScreen(
                    frame,
                    screen_index,
                )
                for screen_index, _
                in enumerate(FRAME_ANNOTATIONS[video.uri][frame.number].screens)
            ] for frame in video._frames
        } for video in VIDEOS.values()
    }


def get_videos():
    """Returns all videos from a XML dataset.

    Returns
    -------
    videos : dict of (str, AnnotatedSampledVideo)
        A map between video file URIs, and all videos from a XML dataset.
    """
    return VIDEOS


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
        A map between page keys, and human annotations associated with the pages of the document.

    Attributes
    ----------
    filename : str
        The filename of the corresponding PDF document. The filename is unique in the video.
    pages : dict of (str, _PageAnnotations)
        A map between page keys, and human annotations associated with the pages of the document.
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
        256-dimensional feature vectors obtained by feeding the page image data into VGG ConvNets.

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
        256-dimensional feature vectors obtained by feeding the page image data into VGG ConvNets.
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
        256-dimensional feature vectors obtained by feeding the page image data into VGG ConvNets.
    """

    def __init__(self, document, key):
        self._document = document
        self.key = key

        page_annotations = DOCUMENT_ANNOTATIONS[document.video.uri][document.filename].pages[key]
        number = page_annotations.number
        self.filename = page_annotations.filename
        self.vgg256 = page_annotations.vgg256

        self._page = ImageFileDocumentPage(document, number, self.pathname)

    @property
    def document(self):
        return self._document

    @property
    def number(self):
        return self._page.number

    @property
    def pathname(self):
        pathname = os.path.join(
            self.document.video.pathname,
            self.filename,
        )
        return pathname

    def image(self, width, height):
        return self._page.image(width, height)


class AnnotatedSampledVideoDocument(DocumentABC):
    """A sequence of images forming a document extracted from a dataset with XML human annotations.

    .. _RFC3987: https://tools.ietf.org/html/rfc3987

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
    uri : string
        An IRI, as defined in RFC3987_, that uniquely indentifies the document over the entire
        lifetime of a program.
    """

    def __init__(self, video, filename):
        self.video = video
        self.filename = filename
        self._pathname = os.path.join(video.pathname, filename)
        self._uri = Path(self._pathname).as_uri()

        document_annotations = DOCUMENT_ANNOTATIONS[video.uri][filename]
        self._pages = sorted([
            AnnotatedSampledVideoDocumentPage(
                self,
                page_annotations.key,
            ) for page_annotations in document_annotations.pages.values()
        ])

    @property
    def title(self):
        return None

    @property
    def author(self):
        return None

    @property
    def pathname(self):
        return self._pathname

    @property
    def uri(self):
        return self._uri

    def __iter__(self):
        return iter(self._pages)

    def __hash__(self):
        return hash(self.pathname)

    def __eq__(self, other):
        if isinstance(other, (AnnotatedSampledVideoDocument, PDFDocument)):
            return self.pathname == other.pathname
        return NotImplemented


class _FrameAnnotations(object):
    """Human annotations associated with a single frame of a video.

    Parameters
    ----------
    filename : str
        The filename of the corresponding video frame image. The filename is unique in the video.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1. The frame number is unique in the video.
    screens : list of _ScreenAnnotations
        A list of human annotations associated with the lit projection screens in the frame.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the frame image data into VGG ConvNets.

    Attributes
    ----------
    filename : str
        The filename of the corresponding video frame image. The filename is unique in the video.
    number : int
        The frame number, i.e. the position of the frame in the video. Frame indexing is one-based,
        i.e. the first frame has number 1. The frame number is unique in the video.
    screens : list of _ScreenAnnotations
        A list of human annotations associated with the lit projection screens in the frame.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the frame image data into VGG ConvNets.
    """

    def __init__(self, filename, number, screens, vgg256):
        self.filename = filename
        self.number = number
        self.screens = screens
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
    image : ndarray
        The image data of the frame as an OpenCV CV_8UC3 RGBA matrix, where the alpha channel (A)
        is currently unused and all pixels are fully opaque, i.e. they have the maximum alpha of
        255.
    width : int
        The width of the image data.
    height : int
        The height of the image data.
    datetime : aware datetime
        The date, and time at which the frame was captured.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the frame image data into VGG ConvNets.
    """

    def __init__(self, video, number):
        self._video = video

        frame_annotations = FRAME_ANNOTATIONS[video.uri][number]
        self.filename = frame_annotations.filename
        self.vgg256 = frame_annotations.vgg256

        bgr_frame_image = cv.imread(self.pathname)
        rgba_frame_image = cv.cvtColor(bgr_frame_image, cv.COLOR_BGR2RGBA)
        self._frame = ImageFrame(video, number, rgba_frame_image)

    @property
    def video(self):
        return self._video

    @property
    def number(self):
        return self._frame.number

    @property
    def pathname(self):
        pathname = os.path.join(
            self.video.pathname,
            self.filename,
        )
        return pathname

    @property
    def image(self):
        return self._frame.image


class _VideoAnnotations(object):
    """Human annotations associated with a single video.

    Parameters
    ----------
    uri : str
        The URI of the video file. The URI is unique in the dataset.
    dirname : str
        The pathname of the directory, where the frames, documents, and XML human annotations
        associated with the video are stored.
    num_frames : int
        The total number of frames in the original video file.
    fps : scalar
        The framerate of the video in frames per second.
    width : int
        The width of the video.
    height : int
        The height of the video.

    Attributes
    ----------
    uri : str
        The URI of the video file. The URI is unique in the dataset.
    dirname : str
        The pathname of the directory, where the frames, documents, and XML human annotations
        associated with the video are stored.
    num_frames : int
        The total number of frames in the original video file.
    fps : scalar
        The framerate of the video in frames per second.
    width : int
        The width of the video.
    height : int
        The height of the video.
    """

    def __init__(self, uri, dirname, fps, num_frames, width, height):
        self.uri = uri
        self.dirname = dirname
        self.num_frames = num_frames
        self.fps = fps
        self.width = width
        self.height = height


class AnnotatedSampledVideo(VideoABC):
    """A sample of a video file extracted from a dataset with XML human annotations.

    Parameters
    ----------
    uri : str
        The URI of the video file. The URI is unique in the dataset.

    Attributes
    ----------
    dirname : str
        The pathname of the directory, where the frames, documents, and XML human annotations
        associated with the video are stored.
    pathname : str
        The full pathname of the directory, where the frames, documents, and XML human annotations
        associated with the video are stored.
    filename : str
        The filename of the video file.
    num_frames : int
        The total number of frames in the original video file.
    fps : scalar
        The framerate of the video in frames per second.
    width : int
        The width of the video.
    height : int
        The height of the video.
    datetime : aware datetime
        The date, and time at which the video was captured.
    documents : dict of (str, AnnotatedSampledVideoDocument)
        A map between PDF document filenames, and the documents associated with the video.
    uri : string
        The URI of the video file. The URI is unique in the dataset.
    """

    def __init__(self, uri):
        self._uri = uri
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
        self.dirname = video_annotations.dirname
        self.num_frames = video_annotations.num_frames
        self._fps = video_annotations.fps
        self._width = video_annotations.width
        self._height = video_annotations.height

        self._frames = sorted([
            AnnotatedSampledVideoFrame(
                self,
                frame_annotations.number
            ) for frame_annotations in FRAME_ANNOTATIONS[uri].values()
        ])

        self.documents = {
            document_annotations.filename: AnnotatedSampledVideoDocument(
                self,
                document_annotations.filename,
            ) for document_annotations in DOCUMENT_ANNOTATIONS[uri].values()
        }

    @property
    def pathname(self):
        pathname = os.path.join(
            RESOURCES_PATHNAME,
            self.dirname,
        )
        return pathname

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

    @property
    def uri(self):
        return self._uri

    def __iter__(self):
        return iter(self._frames)


class _KeyRefAnnotations(object):
    """Human annotations describing a document page shown in a lit projection screen.

    Parameters
    ----------
    key : str
        An identifier of a page in a document. The identifier is unique in the video associated with
        the document.
    similarity : str
        The similarity between  what is shown in the projection screen, and the document page. The
        following values are legal:

        - `full` specifies that there is a 1:1 correspondence between what is shown in the
          projection screen, and the document page.
        - `incremental` specifies that in a document attached to the ancestor video, a single
          logical page is split across multiple physical pages and incrementally uncovered; the
          slide and the frame correspond to the same logical page, but not the same physical page.

    Attributes
    ----------
    key : str
        An identifier of a page in a document. The identifier is unique in the video associated with
        the document.
    similarity : str
        The similarity between  what is shown in the projection screen, and the document page. The
        following values are legal:

        - `full` specifies that there is a 1:1 correspondence between what is shown in the
          projection screen, and the document page.
        - `incremental` specifies that in a document attached to the ancestor video, a single
          logical page is split across multiple physical pages and incrementally uncovered; the
          slide and the frame correspond to the same logical page, but not the same physical page.

    """

    def __init__(self, key, similarity):
        self.key = key
        self.similarity = similarity


class _ScreenAnnotations(object):
    """Human annotations associated with a single lit projection screen in a frame of a video.

    Parameters
    ----------
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    condition : str
        The condition of what is being shown in the screen. The following values are legal:

        - `pristine` specifies that there is no significant degradation beyond photon noise.
        - `windowed` specifies that a slide is being shown, but the slide does not cover the full
          screen.
        - `obstacle` specifies that a part of the screen or the projector light is partially
          obscured by either a physical obstacle, or by a different GUI window.

    keyrefs : dict of (str, _KeyRefAnnotations)
        A map between document page keys, and human annotations specifying the relationship between
        the projection screen, and the document pages.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the screen image data into VGG ConvNets.

    Attributes
    ----------
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    condition : str
        The condition of what is being shown in the screen. The following values are legal:

        - `pristine` specifies that there is no significant degradation beyond photon noise.
        - `windowed` specifies that a slide is being shown, but the slide does not cover the full
          screen.
        - `obstacle` specifies that a part of the screen or the projector light is partially
          obscured by either a physical obstacle, or by a different GUI window.

    keyrefs : dict of (str, _KeyRefAnnotations)
        A map between document page keys, and human annotations specifying the relationship between
        the projection screen, and the document pages.
    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the screen image data into VGG ConvNets.
    """

    def __init__(self, coordinates, condition, keyrefs, vgg256):
        self.coordinates = coordinates
        self.condition = condition
        self.keyrefs = keyrefs
        self.vgg256 = vgg256


class AnnotatedSampledVideoScreen(ScreenABC):
    """A projection screen extracted from XML human annotations.

    Parameters
    ----------
    frame : FrameABC
        A video frame containing the projection screen.
    screen_index : int
        The index of the projection screen in the human annotations for the video frame. Screen
        indexing is zero-based, i.e. the first screen in the human annotations has index 0.

    Attributes
    ----------
    frame : FrameABC
        A video frame containing the projection screen.
    coordinates : ConvexQuadrangleABC
        A map between frame and screen coordinates.
    condition : str
        The condition of what is being shown in the screen. The following values are legal:

        - `pristine` specifies that there is no significant degradation beyond photon noise.
        - `windowed` specifies that a slide is being shown, but the slide does not cover the full
          screen.
        - `obstacle` specifies that a part of the screen or the projector light is partially
          obscured by either a physical obstacle, or by a different GUI window.

    vgg256 : VGG256Features
        256-dimensional feature vectors obtained by feeding the screen image data into VGG ConvNets.
    """

    def __init__(self, frame, screen_index):
        self._frame = frame
        self._screen_index = screen_index

        screen_annotations = FRAME_ANNOTATIONS[frame.video.uri][frame.number].screens[screen_index]
        self._coordinates = screen_annotations.coordinates
        self.condition = screen_annotations.condition
        self.vgg256 = screen_annotations.vgg256

    @property
    def frame(self):
        return self._frame

    @property
    def coordinates(self):
        return self._coordinates

    def matching_pages(self):
        r"""Returns an iterable of document pages matching the screen based on human annotations.

        Note
        ----
        When a projection screen :math:`s` shows a document page :math:`p`, we write :math:`s\approx
        p`.  When a single logical document page is split across several document pages :math:`p`
        and a projection screen :math:`s` shows the same logical page as :math:`p`, we write
        :math:`s\sim p`. A document page :math:`p` is matching a projection screen :math:`s` if and
        only if :math:`s\approx p\lor (\nexists p'(s\approx p') \land s \sim p)`.

        Returns
        -------
        matching_pages : iterable of AnnotatedSampledVideoDocumentPage
            An iterable of document pages matching the projection screen based on human annotations.
        """
        video_uri = self._frame.video.uri
        frame_number = self._frame.number
        frame_annotations = FRAME_ANNOTATIONS[video_uri][frame_number]
        screen_index = self._screen_index
        screen_annotations = frame_annotations.screens[screen_index]
        keyrefs = screen_annotations.keyrefs.values()
        pages = PAGES[self._frame.video.uri]
        full_matches = [pages[keyref.key] for keyref in keyrefs if keyref.similarity == 'full']
        if full_matches:
            return full_matches
        incremental_matches = [pages[keyref.key] for keyref in keyrefs]
        return incremental_matches


class AnnotatedSampledVideoScreenDetector(ScreenDetectorABC):
    """A screen detector that maps an annotated video frame to screens using XML human annotations.

    Parameters
    ----------
    conditions : set of str, optional
        A set of admissible conditions of a screen. The following condition strings are legal:

        - `pristine` specifies that there is no significant degradation beyond photon noise.
        - `windowed` specifies that a slide is being shown, but the slide does not cover the full
          screen.
        - `obstacle` specifies that a part of the screen or the projector light is partially
          obscured by either a physical obstacle, or by a different GUI window.

        Screens with inadmissible conditions will not be detected.
    """

    def __init__(self, conditions=set(['pristine', 'windowed', 'obstacle'])):
        self._conditions = conditions

    def detect(self, frame):
        """Converts an annotated frame to screens using the closest available human annotations.

        Parameters
        ----------
        frame : FrameABC
            A frame of a video.

        Returns
        -------
        screens : iterable of AnnotatedSampledVideoScreen
            An iterable of detected lit projection screens.
        """
        if not isinstance(frame, AnnotatedSampledVideoFrame):
            return ()
        return [
            screen for screen in SCREENS[frame.video.uri][frame.number]
            if screen.condition in self._conditions
        ]


_init_dataset()
