# -*- coding: utf-8 -*-

import os
import unittest

import cv2 as cv

from video699.document.image_file import ImageFileDocument


RESOURCES_PATHNAME = os.path.join(os.path.dirname(__file__), 'test_image_file')
FIRST_PAGE_IMAGE_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'sample_pdf_document_first_page.png')
SECOND_PAGE_IMAGE_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'sample_pdf_document_second_page.png')
PAGE_IMAGE_WIDTH = 2550
PAGE_IMAGE_HEIGHT = 3300
PAGE_IMAGE_HORIZONTAL_MARGIN = 100
PAGE_IMAGE_VERTICAL_MARGIN = 200


class TestImageFileDocumentPage(unittest.TestCase):
    """Tests the ability of the ImageFileDocumentPage class to read images and produce page numbers.

    """

    def setUp(self):
        image_pathnames = (
            FIRST_PAGE_IMAGE_PATHNAME,
            SECOND_PAGE_IMAGE_PATHNAME,
        )
        self.document = ImageFileDocument(image_pathnames)
        page_iterator = iter(self.document)
        self.first_page = next(page_iterator)
        self.second_page = next(page_iterator)

    def test_page_number(self):
        self.assertEqual(1, self.first_page.number)
        self.assertEqual(2, self.second_page.number)

    def test_first_page_image(self):
        image = self.first_page.render(PAGE_IMAGE_WIDTH, PAGE_IMAGE_HEIGHT)
        height, width, _ = image.shape
        self.assertEqual(PAGE_IMAGE_WIDTH, width)
        self.assertEqual(PAGE_IMAGE_HEIGHT, height)

        red, green, blue, alpha = cv.split(image)

        position = (0, 0)
        self.assertEqual(255, blue[position])
        self.assertEqual(255, green[position])
        self.assertEqual(255, red[position])
        self.assertEqual(255, alpha[position])

        position = (675, 900)
        self.assertEqual(0, blue[position])
        self.assertEqual(0, green[position])
        self.assertEqual(255, red[position])
        self.assertEqual(255, alpha[position])

        position = (2200, 900)
        self.assertEqual(255, blue[position])
        self.assertEqual(255, green[position])
        self.assertEqual(255, red[position])
        self.assertEqual(255, alpha[position])

        position = (2200, 1200)
        self.assertEqual(0, blue[position])
        self.assertEqual(255, green[position])
        self.assertEqual(0, red[position])
        self.assertEqual(255, alpha[position])

    def test_second_page_image(self):
        image = self.second_page.render(PAGE_IMAGE_WIDTH, PAGE_IMAGE_HEIGHT)
        height, width, _ = image.shape
        self.assertEqual(PAGE_IMAGE_WIDTH, width)
        self.assertEqual(PAGE_IMAGE_HEIGHT, height)

        red, green, blue, alpha = cv.split(image)

        position = (0, 0)
        self.assertEqual(255, blue[position])
        self.assertEqual(255, green[position])
        self.assertEqual(255, red[position])
        self.assertEqual(255, alpha[position])

        position = (1250, 900)
        self.assertEqual(255, blue[position])
        self.assertEqual(255, green[position])
        self.assertEqual(255, red[position])
        self.assertEqual(255, alpha[position])

        position = (1250, 1200)
        self.assertEqual(255, blue[position])
        self.assertEqual(0, green[position])
        self.assertEqual(0, red[position])
        self.assertEqual(255, alpha[position])

    def test_wider_aspect_ratio(self):
        image = self.first_page.render(
            PAGE_IMAGE_WIDTH + PAGE_IMAGE_HORIZONTAL_MARGIN,
            PAGE_IMAGE_HEIGHT,
        )
        height, width, _ = image.shape
        self.assertEqual(PAGE_IMAGE_WIDTH + PAGE_IMAGE_HORIZONTAL_MARGIN, width)
        self.assertEqual(PAGE_IMAGE_HEIGHT, height)

        *_, alpha = cv.split(image)
        screen_corners = (
            (0, 0),
            (0, width - 1),
            (int((height - 1) / 2), 0),
            (int((height - 1) / 2), 0),
            (height - 1, 0),
            (height - 1, width - 1),
        )
        for coordinates in screen_corners:
            self.assertEqual(0, alpha[coordinates])

    def test_taller_aspect_ratio(self):
        image = self.first_page.render(
            PAGE_IMAGE_WIDTH,
            PAGE_IMAGE_HEIGHT + PAGE_IMAGE_VERTICAL_MARGIN,
        )
        height, width, _ = image.shape
        self.assertEqual(PAGE_IMAGE_WIDTH, width)
        self.assertEqual(PAGE_IMAGE_HEIGHT + PAGE_IMAGE_VERTICAL_MARGIN, height)

        *_, alpha = cv.split(image)
        screen_corners = (
            (0, 0),
            (0, int((width - 1) / 2)),
            (0, width - 1),
            (height - 1, 0),
            (height - 1, int((width - 1) / 2)),
            (height - 1, width - 1),
        )
        for coordinates in screen_corners:
            self.assertEqual(0, alpha[coordinates])


if __name__ == '__main__':
    unittest.main()
