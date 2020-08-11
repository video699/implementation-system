# -*- coding: utf-8 -*-

"""This module provides the main script of the video699 package.

"""

import argparse
from glob import glob
from itertools import chain  # FIXME

from dateutil.parser import parse
from lxml.etree import xmlfile

from .interface import DocumentABC, VideoABC, ConvexQuadrangleTrackerABC, ScreenDetectorABC, \
    PageDetectorABC
from .event.screen import ScreenEventDetectorABC


QUADRANGLE_TRACKER_NAMES = ['rtree_deque']
SCREEN_DETECTOR_NAMES = ['fastai', 'annotated']
SCENE_DETECTOR_NAMES = ['distance', 'none']
PAGE_DETECTOR_NAMES = ['siamese', 'imagehash', 'vgg16', 'annotated']


def _documents(args):
    """Reads documents specified by the arguments of the main script.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments received by the main script.

    Returns
    -------
    documents : set of DocumentABC
        Documents specified by the arguments of the main script.
    """

    documents = set()
    for uri in args.documents:
        if uri.lower().endswith('.pdf'):
            from .document.pdf import PDFDocument
            document = PDFDocument(uri)
        else:
            from .document.image_file import ImageFileDocument
            uris = sorted(glob('{}/*'.format(uri)))
            if not uris:
                raise ValueError('{} is not a directory or is empty'.format(uri))
            document = ImageFileDocument(uris)
        assert isinstance(document, DocumentABC)
        documents.add(document)
    return documents


def _video(args):
    """Reads a video specified by the arguments of the main script.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments received by the main script.

    Returns
    -------
    video : VideoABC
        A video specified by the arguments of the main script.
    """

    uri = args.video
    from .video.file import VideoFile
    date = args.date
    if date is None:
        raise ValueError('Video requires capture date')
    video = VideoFile(
        pathname=uri,
        datetime=parse(date),
        verbose=True,
    )
    assert isinstance(video, VideoABC)
    return video


def _scene_detector(video, args):
    """Produces a scene event detector from the arguments of the main script.

    Parameters
    ----------
    video : VideoABC
        A video in which the scene detector will detect important frames.
    args : argparse.Namespace
        The arguments received by the main script.

    Returns
    -------
    scene_event_detector : VideoABC
        The scene event detector from the arguments of the main script.
    """

    assert isinstance(video, VideoABC)
    name = args.scene_detector
    assert name in SCENE_DETECTOR_NAMES
    if name == 'distance':
        from .video.scene import MeanSquaredErrorSceneDetector
        scene_detector = MeanSquaredErrorSceneDetector(video)
    elif name == 'none':
        scene_detector = video
    assert isinstance(scene_detector, VideoABC)
    return scene_detector


def _screen_event_detector(args):
    """Produces a screen event detector from the arguments of the main script.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments received by the main script.

    Returns
    -------
    screen_event_detector : ScreenEventDetectorABC
        The screen event detector from the arguments of the main script.
    """

    convex_quadrangle_tracker = _convex_quadrangle_tracker(args)
    screen_detector = _screen_detector(args)
    page_detector = _page_detector(args)
    video = _scene_detector(_video(args), args)

    from .event.screen import ScreenEventDetector
    screen_event_detector = ScreenEventDetector(
        video,
        convex_quadrangle_tracker,
        screen_detector,
        page_detector,
    )
    assert isinstance(screen_event_detector, ScreenEventDetectorABC)
    return screen_event_detector


def _convex_quadrangle_tracker(args):
    """Produces a convex quadrangle tracker from the arguments of the main script.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments received by the main script.

    Returns
    -------
    convex_quadrangle_tracker : ConvexQuadrangleTrackerABC
        The convex quadrangle tracker from the arguments of the main script.
    """

    name = args.convex_quadrangle_tracker
    assert name in QUADRANGLE_TRACKER_NAMES
    if name == 'rtree_deque':
        from .quadrangle.rtree import RTreeDequeConvexQuadrangleTracker
        convex_quadrangle_tracker = RTreeDequeConvexQuadrangleTracker(2)
    assert isinstance(convex_quadrangle_tracker, ConvexQuadrangleTrackerABC)
    return convex_quadrangle_tracker


def _screen_detector(args):
    """Produces a screen detector from the arguments of the main script.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments received by the main script.

    Returns
    -------
    screen_detector : ScreenDetectorABC
        The screen detector from the arguments of the main script.
    """

    name = args.screen_detector
    assert name in SCREEN_DETECTOR_NAMES
    if name == 'fastai':
        from .screen.semantic_segmentation.fastai_detector import FastAIScreenDetector
        screen_detector = FastAIScreenDetector()
    elif name == 'annotated':
        institution_id = args.institution
        room_id = args.room
        camera_id = args.camera
        if institution_id is None or room_id is None or camera_id is None:
            raise ValueError('Annotated screen detector requires institution, room, and camera IDs')
        from .screen.annotated import AnnotatedScreenDetector
        screen_detector = AnnotatedScreenDetector(
            institution_id=institution_id,
            room_id=room_id,
            camera_id=camera_id,
        )
    assert isinstance(screen_detector, ScreenDetectorABC)
    return screen_detector


def _page_detector(args):
    """Produces a page detector from the arguments of the main script.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments received by the main script.

    Returns
    -------
    page_detector : PageDetectorABC
        The page detector from the arguments of the main script.
    """

    name = args.page_detector
    assert name in PAGE_DETECTOR_NAMES
    if name == 'siamese':
        from .page.siamese import KerasSiamesePageDetector
        page_detector = KerasSiamesePageDetector(_documents(args))
    elif name == 'imagehash':
        from video699.page.imagehash import ImageHashPageDetector
        page_detector = ImageHashPageDetector(_documents(args))
    elif name == 'vgg16':
        from video699.page.vgg16 import KerasVGG16PageDetector
        page_detector = KerasVGG16PageDetector(_documents(args))
    elif name == 'annotated':  # FIXME
        page_detector = AnnotatedPageDetector(_documents(args))
    assert isinstance(page_detector, PageDetectorABC)
    return page_detector


class AnnotatedPageDetector(PageDetectorABC):  # FIXME
    def __init__(self, documents):
        document = list(next(iter(documents)))
        assert len(document) == 65
        self.document = document

    def detect(self, frame, appeared_screens, existing_screens, disappeared_screens):
        document = self.document
        if frame.number < 693:
            detected_page = document[0]
        elif frame.number < 1462:
            detected_page = document[1]
        elif frame.number < 1491:
            detected_page = document[2]
        elif frame.number < 1755:
            detected_page = document[3]
        elif frame.number < 2371:
            detected_page = document[4]
        elif frame.number < 2913:
            detected_page = document[5]
        elif frame.number < 4174:
            detected_page = document[6]
        elif frame.number < 4230:
            detected_page = document[7]
        elif frame.number < 4598:
            detected_page = document[8]
        elif frame.number < 5058:
            detected_page = document[9]
        elif frame.number < 5348:
            detected_page = document[10]
        elif frame.number < 5910:
            detected_page = document[11]
        elif frame.number < 6362:
            detected_page = document[12]
        elif frame.number < 7023:
            detected_page = document[13]
        elif frame.number < 7733:
            detected_page = document[14]
        elif frame.number < 8293:
            detected_page = document[15]
        elif frame.number < 9996:
            detected_page = document[16]
        elif frame.number < 11596:
            detected_page = document[17]
        elif frame.number < 11707:
            detected_page = document[18]
        elif frame.number < 12034:
            detected_page = document[19]
        elif frame.number < 12259:
            detected_page = document[20]
        elif frame.number < 13308:
            detected_page = document[21]
        elif frame.number < 13902:
            detected_page = document[22]
        elif frame.number < 14339:
            detected_page = document[23]
        elif frame.number < 14339:
            detected_page = document[23]
        elif frame.number < 14721:
            detected_page = document[24]
        elif frame.number < 15125:
            detected_page = document[25]
        elif frame.number < 15605:
            detected_page = document[26]
        elif frame.number < 15714:
            detected_page = document[27]
        elif frame.number < 16094:
            detected_page = document[28]
        elif frame.number < 16094:
            detected_page = document[28]
        elif frame.number < 17714:
            detected_page = document[29]
        elif frame.number < 18499:
            detected_page = document[30]
        elif frame.number < 18716:
            detected_page = document[31]
        elif frame.number < 19439:
            detected_page = document[32]
        elif frame.number < 20251:
            detected_page = document[33]
        elif frame.number < 21404:
            detected_page = document[34]
        elif frame.number < 21803:
            detected_page = document[35]
        elif frame.number < 22689:
            detected_page = document[36]
        elif frame.number < 22909:
            detected_page = document[37]
        elif frame.number < 23239:
            detected_page = document[38]
        elif frame.number < 23439:
            detected_page = document[39]
        elif frame.number < 23714:
            detected_page = document[40]
        elif frame.number < 24005:
            detected_page = document[41]
        elif frame.number < 24038:
            detected_page = document[42]
        elif frame.number < 25012:
            detected_page = document[43]
        elif frame.number < 25342:
            detected_page = document[44]
        elif frame.number < 26361:
            detected_page = document[45]
        elif frame.number < 27868:
            detected_page = document[46]
        elif frame.number < 27903:
            detected_page = document[47]
        elif frame.number < 41622:
            detected_page = None
        elif frame.number < 41751:
            detected_page = document[47]
        elif frame.number < 42259:
            detected_page = document[48]
        elif frame.number < 43056:
            detected_page = document[49]
        elif frame.number < 43864:
            detected_page = document[50]
        elif frame.number < 44706:
            detected_page = document[51]
        elif frame.number < 45127:
            detected_page = document[52]
        elif frame.number < 47954:
            detected_page = document[53]
        elif frame.number < 48666:
            detected_page = document[54]
        elif frame.number < 48913:
            detected_page = document[53]
        elif frame.number < 49383:
            detected_page = document[54]
        elif frame.number < 49724:
            detected_page = document[53]
        elif frame.number < 50281:
            detected_page = document[54]
        elif frame.number < 50387:
            detected_page = document[53]
        elif frame.number < 50403:
            detected_page = document[54]
        elif frame.number < 50415:
            detected_page = document[55]
        elif frame.number < 50540:
            detected_page = document[56]
        elif frame.number < 50605:
            detected_page = document[57]
        elif frame.number < 52012:
            detected_page = document[58]
        elif frame.number < 55419:
            detected_page = document[59]
        elif frame.number < 56658:
            detected_page = document[60]
        elif frame.number < 56802:
            detected_page = document[61]
        elif frame.number < 58516:
            detected_page = document[62]
        elif frame.number < 58536:
            detected_page = document[61]
        elif frame.number < 58561:
            detected_page = document[62]
        elif frame.number < 60096:
            detected_page = document[63]
        elif frame.number < 63672:
            detected_page = document[64]
        else:
            detected_page = None
        detected_pages = dict()
        for screen, _ in chain(appeared_screens, existing_screens):
            detected_pages[screen] = detected_page
        return detected_pages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Aligns lecture recording with study materials.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-c',
        '--convex-quadrangle-tracker',
        default='rtree_deque',
        help=(
            'the convex quadrangle tracker that will be used to track the movement of lit'
            ' projection screens in the video'
        ),
        choices=QUADRANGLE_TRACKER_NAMES,
    )
    parser.add_argument(
        '-s',
        '--screen-detector',
        default='annotated',
        help=(
            'the screen detector that will be used to detect the location of lit projection screens'
            ' in the video'
        ),
        choices=SCREEN_DETECTOR_NAMES,
    )
    parser.add_argument(
        '-S',
        '--scene-detector',
        default='none',
        help=(
            'the scene detector that will be used to detect important frames in the video'
        ),
        choices=SCENE_DETECTOR_NAMES,
    )
    parser.add_argument(
        '-p',
        '--page-detector',
        default='annotated',
        help=(
            'the page detector that will be used to detect document pages in lit projection screens'
            ' in the video'
        ),
        choices=PAGE_DETECTOR_NAMES,
    )
    parser.add_argument(
        '-i',
        '--institution',
        default=None,
        help='the ID of the institution at which the video was captured',
    )
    parser.add_argument(
        '-r',
        '--room',
        default=None,
        help='the ID of the room at which the video was captured',
    )
    parser.add_argument(
        '-C',
        '--camera',
        default=None,
        help='the ID of the camera with which the video was captured',
    )
    parser.add_argument(
        '-D',
        '--date',
        default=None,
        help='the date at which the video was captured',
    )
    parser.add_argument(
        '-d',
        '--documents',
        metavar='DOCUMENT',
        help='the study material documents whose pages will be considered by the page detector',
        required=True,
        nargs='+',
    )
    parser.add_argument(
        '-v',
        '--video',
        help='the video in which the screen detector will detect lit projection screens',
        required=True,
    )
    parser.add_argument(
        '-o',
        '--output',
        help='the output produced from the video and the study materials',
        required=True,
    )

    args = parser.parse_args()
    event_detector = _screen_event_detector(args)
    with xmlfile(args.output, encoding='utf-8') as xf:
        event_detector.write_xml(xf)
