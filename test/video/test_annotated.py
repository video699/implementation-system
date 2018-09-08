# -*- coding: utf-8 -*-

import unittest

from dateutil.parser import parse as datetime_parse
from video699.video.annotated import get_videos


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
