# -*- coding: utf-8 -*-

import unittest

import cv2 as cv
from dateutil.parser import parse as datetime_parse
from video699.video.annotated import get_videos, AnnotatedSampledVideoScreenDetector


VIDEOS = get_videos()
VIDEO_URI = 'https://is.muni.cz/auth/el/{faculty}/{term}/{course}/um/vi/?videomuni={fname}'.format(
    course='PB029',
    faculty=1433,
    fname='PB029-D3-20161026.mp4',
    term='podzim2016',
)
VIDEO_DIRNAME = 'PB029-D3-20161026.mp4'
VIDEO_NUM_FRAMES = 90378
VIDEO_FPS = 15
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 576
VIDEO_DATETIME = datetime_parse('2016-10-26T00:00:00+00:00')
VIDEO_NUM_DOCUMENTS = 4
VIDEO_FRAME_NUMBERS = (
    2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 24000, 30000, 40000, 60000,
    62000, 64000, 66000, 68000, 78000, 80000, 82000, 84000, 86000, 88000, 90000,
)
VGG256_SHAPE = (256,)
FIRST_DOCUMENT_FILENAME = 'slides01.pdf'
SECOND_DOCUMENT_FILENAME = 'slides02.pdf'
FIRST_DOCUMENT_NUM_PAGES = 32
SECOND_DOCUMENT_NUM_PAGES = 10
FIRST_SCREEN_WIDTH = 369
FIRST_SCREEN_HEIGHT = 283


class TestAnnotatedSampledVideo(unittest.TestCase):
    """Tests the ability of the AnnotatedSampledVideo class to read human annotations.

    """

    def setUp(self):
        self.video = VIDEOS[VIDEO_URI]

    def test_video_properties(self):
        self.assertEqual(VIDEO_DIRNAME, self.video.dirname)
        self.assertEqual(VIDEO_NUM_FRAMES, self.video.num_frames)
        self.assertEqual(VIDEO_FPS, self.video.fps)
        self.assertEqual(VIDEO_WIDTH, self.video.width)
        self.assertEqual(VIDEO_HEIGHT, self.video.height)
        self.assertEqual(VIDEO_DATETIME, self.video.datetime)

    def test_video_contains_n_documents(self):
        self.assertEqual(VIDEO_NUM_DOCUMENTS, len(self.video.documents))

    def test_video_produces_n_frames(self):
        self.assertEqual(len(VIDEO_FRAME_NUMBERS), len(list(iter(self.video))))


class TestAnnotatedSampledVideoFrame(unittest.TestCase):
    """Tests the ability of the AnnotatedSampledVideoFrame class to read human annotations.

    """

    def setUp(self):
        video = VIDEOS[VIDEO_URI]
        frame_iterator = iter(video)
        self.first_frame = next(frame_iterator)
        self.second_frame = next(frame_iterator)

    def test_frame_numbers(self):
        self.assertTrue(VIDEO_FRAME_NUMBERS[0], self.first_frame.number)
        self.assertTrue(VIDEO_FRAME_NUMBERS[1], self.second_frame.number)

    def test_frame_image(self):
        frame_image = self.first_frame.image
        height, width, _ = frame_image.shape
        self.assertEqual(VIDEO_WIDTH, width)
        self.assertEqual(VIDEO_HEIGHT, height)

        blue, green, red = cv.split(frame_image)
        self.assertTrue(red[90, 490] > blue[90, 490] and red[90, 490] > green[90, 490])
        self.assertTrue(green[190, 340] > red[190, 340] and green[190, 340] > blue[190, 340])
        self.assertTrue(blue[50, 320] > red[50, 320] and blue[50, 320] > green[50, 320])

    def test_vgg256_dimensions(self):
        self.assertEqual(VGG256_SHAPE, self.first_frame.vgg256.imagenet.shape)
        self.assertEqual(VGG256_SHAPE, self.first_frame.vgg256.imagenet_and_places2.shape)
        self.assertEqual(VGG256_SHAPE, self.second_frame.vgg256.imagenet.shape)
        self.assertEqual(VGG256_SHAPE, self.second_frame.vgg256.imagenet_and_places2.shape)


class TestAnnotatedSampledVideoDocument(unittest.TestCase):
    """Tests the ability of the AnnotatedSampledVideoDocument class to read human annotations.

    """

    def setUp(self):
        video = VIDEOS[VIDEO_URI]
        self.first_document = video.documents[FIRST_DOCUMENT_FILENAME]
        self.second_document = video.documents[SECOND_DOCUMENT_FILENAME]

    def test_reads_n_pages(self):
        self.assertEqual(FIRST_DOCUMENT_NUM_PAGES, len(list(self.first_document)))
        self.assertEqual(SECOND_DOCUMENT_NUM_PAGES, len(list(self.second_document)))


class TestAnnotatedSampledVideoDocumentPage(unittest.TestCase):
    """Tests the ability of the AnnotatedSampledVideoDocumentPage class to read human annotations.

    """

    def setUp(self):
        video = VIDEOS[VIDEO_URI]
        document = video.documents[FIRST_DOCUMENT_FILENAME]
        page_iterator = iter(document)
        self.first_page = next(page_iterator)
        self.second_page = next(page_iterator)

    def test_page_numbers(self):
        self.assertTrue(1, self.first_page.number)
        self.assertTrue(2, self.second_page.number)

    def test_frame_image(self):
        page_width = 640
        page_height = 480
        page_image = self.second_page.image(page_width, page_height)
        height, width, _ = page_image.shape
        self.assertEqual(page_width, width)
        self.assertEqual(page_height, height)

        blue, green, red = cv.split(page_image)

        self.assertEqual(255, blue[50, 260])
        self.assertEqual(0, green[50, 260])
        self.assertEqual(0, red[50, 260])

        self.assertEqual(0, blue[270, 300])
        self.assertEqual(127, green[270, 300])
        self.assertEqual(0, red[270, 300])

        self.assertEqual(0, blue[100, 540])
        self.assertEqual(0, green[100, 540])
        self.assertEqual(255, red[100, 540])

    def test_vgg256_dimensions(self):
        self.assertEqual(VGG256_SHAPE, self.first_page.vgg256.imagenet.shape)
        self.assertEqual(VGG256_SHAPE, self.first_page.vgg256.imagenet_and_places2.shape)
        self.assertEqual(VGG256_SHAPE, self.second_page.vgg256.imagenet.shape)
        self.assertEqual(VGG256_SHAPE, self.second_page.vgg256.imagenet_and_places2.shape)


class TestAnnotatedSampledVideoScreen(unittest.TestCase):
    """Tests the ability of the AnnotatedSampledVideoScreen class to read human annotations.

    """

    def setUp(self):
        video = VIDEOS[VIDEO_URI]
        frames = list(video)
        screen_detector = AnnotatedSampledVideoScreenDetector()
        first_frame = frames[0]
        eleventh_frame = frames[10]
        self.first_screen = next(iter(screen_detector(first_frame)))
        self.second_screen = next(iter(screen_detector(eleventh_frame)))

    def test_matching_pages(self):
        self.assertEqual(
            set([
                page.key
                for page in self.first_screen.matching_pages()
            ]), set([
                'slides01-02',
            ])
        )
        self.assertEqual(
            set([
                page.key
                for page in self.second_screen.matching_pages()
            ]), set([
                'slides02-04',
            ])
        )

    def test_screen_image(self):
        screen_image = self.first_screen.image
        height, width, _ = screen_image.shape
        self.assertTrue(width > height)

        blue, green, red = cv.split(screen_image)

        coordinates = (int(0.23 * height), int(0.85 * width))
        self.assertTrue(red[coordinates] > green[coordinates])
        self.assertTrue(red[coordinates] > blue[coordinates])

        coordinates = (int(0.57 * height), int(0.43 * width))
        self.assertTrue(green[coordinates] > red[coordinates])
        self.assertTrue(green[coordinates] > blue[coordinates])

        coordinates = (int(0.09 * height), int(0.41 * width))
        self.assertTrue(blue[coordinates] > red[coordinates])
        self.assertTrue(blue[coordinates] > green[coordinates])

    def test_vgg256_dimensions(self):
        self.assertEqual(VGG256_SHAPE, self.first_screen.vgg256.imagenet.shape)
        self.assertEqual(VGG256_SHAPE, self.first_screen.vgg256.imagenet_and_places2.shape)
        self.assertEqual(VGG256_SHAPE, self.second_screen.vgg256.imagenet.shape)
        self.assertEqual(VGG256_SHAPE, self.second_screen.vgg256.imagenet_and_places2.shape)
