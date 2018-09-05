# -*- coding: utf-8 -*-

import os
import unittest

import cv2 as cv
from dateutil.parser import parse
from video699.video.file import VideoFile


VIDEO_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'test_file',
    'sample_video_file.mov'
)
VIDEO_FPS = 25
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_DATETIME = parse('2018-01-01T00:00:00+00:00')


class TestVideoFile(unittest.TestCase):
    """Tests the ability of the VideoFile class to read a RLE-encoded video file.

    """

    def test_video_properties(self):
        video = VideoFile(VIDEO_PATHNAME, VIDEO_DATETIME)
        self.assertEqual(VIDEO_FPS, video.fps)
        self.assertEqual(VIDEO_WIDTH, video.width)
        self.assertEqual(VIDEO_HEIGHT, video.height)

    def test_reads_frame(self):
        video = VideoFile(VIDEO_PATHNAME, VIDEO_DATETIME)
        frame_iterator = video.__iter__()
        frame = next(frame_iterator)
        self.assertEqual(VIDEO_WIDTH, frame.width)
        self.assertEqual(VIDEO_HEIGHT, frame.height)

        frame_image = frame.image
        height, width, _ = frame_image.shape
        self.assertEqual(VIDEO_WIDTH, width)
        self.assertEqual(VIDEO_HEIGHT, height)

        blue, green, red = cv.split(frame_image)

        self.assertEqual(0, blue[282, 144])
        self.assertEqual(0, green[282, 144])
        self.assertEqual(0, red[282, 144])

        self.assertEqual(255, blue[205, 508])
        self.assertEqual(0, green[205, 508])
        self.assertEqual(0, red[205, 508])

        self.assertEqual(0, blue[410, 220])
        self.assertEqual(255, green[410, 220])
        self.assertEqual(0, red[410, 220])

        self.assertEqual(0, blue[150, 170])
        self.assertEqual(0, green[150, 170])
        self.assertEqual(255, red[150, 170])

    def test_produces_single_frame(self):
        video = VideoFile(VIDEO_PATHNAME, VIDEO_DATETIME)
        frame_iterator = video.__iter__()
        next(frame_iterator)
        with self.assertRaises(StopIteration):
            next(frame_iterator)


if __name__ == '__main__':
    unittest.main()
