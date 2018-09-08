# -*- coding: utf-8 -*-

import os
import unittest

import cv2 as cv

from video699.document.image import ImageDocument


RESOURCES_PATHNAME = os.path.join(os.path.dirname(__file__), 'test_image')
FIRST_PAGE_IMAGE_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'sample_pdf_document_first_page.png')
SECOND_PAGE_IMAGE_PATHNAME = os.path.join(RESOURCES_PATHNAME, 'sample_pdf_document_second_page.png')
PAGE_IMAGE_WIDTH = 2550
PAGE_IMAGE_HEIGHT = 3300


class TestImageDocumentPage(unittest.TestCase):
    """Tests the ability of the ImageDocumentPage class to rescale images and produce page numbers.

    """

    def setUp(self):
        page_images = (
            cv.imread(FIRST_PAGE_IMAGE_PATHNAME),
            cv.imread(SECOND_PAGE_IMAGE_PATHNAME),
        )
        self.document = ImageDocument(page_images)
        page_iterator = iter(self.document)
        self.first_page = next(page_iterator)
        self.second_page = next(page_iterator)
        self.first_page_image = self.first_page.image(PAGE_IMAGE_WIDTH, PAGE_IMAGE_HEIGHT)
        self.second_page_image = self.second_page.image(PAGE_IMAGE_WIDTH, PAGE_IMAGE_HEIGHT)

    def test_document(self):
        self.assertEqual(self.document, self.first_page.document)
        self.assertEqual(self.document, self.second_page.document)

    def test_page_number(self):
        self.assertEqual(1, self.first_page.number)
        self.assertEqual(2, self.second_page.number)

    def test_first_page_image(self):
        height, width, _ = self.first_page_image.shape
        self.assertEqual(PAGE_IMAGE_WIDTH, width)
        self.assertEqual(PAGE_IMAGE_HEIGHT, height)

        blue, green, red = cv.split(self.first_page_image)

        self.assertEqual(255, blue[0, 0])
        self.assertEqual(255, green[0, 0])
        self.assertEqual(255, red[0, 0])

        self.assertEqual(0, blue[675, 900])
        self.assertEqual(0, green[675, 900])
        self.assertEqual(255, red[675, 900])

        self.assertEqual(255, blue[2200, 900])
        self.assertEqual(255, green[2200, 900])
        self.assertEqual(255, red[2200, 900])

        self.assertEqual(0, blue[2200, 1200])
        self.assertEqual(255, green[2200, 1200])
        self.assertEqual(0, red[2200, 1200])

    def test_second_page_image(self):
        height, width, _ = self.second_page_image.shape
        self.assertEqual(PAGE_IMAGE_WIDTH, width)
        self.assertEqual(PAGE_IMAGE_HEIGHT, height)

        blue, green, red = cv.split(self.second_page_image)

        self.assertEqual(255, blue[0, 0])
        self.assertEqual(255, green[0, 0])
        self.assertEqual(255, red[0, 0])

        self.assertEqual(255, blue[1250, 900])
        self.assertEqual(255, green[1250, 900])
        self.assertEqual(255, red[1250, 900])

        self.assertEqual(255, blue[1250, 1200])
        self.assertEqual(0, green[1250, 1200])
        self.assertEqual(0, red[1250, 1200])


if __name__ == '__main__':
    unittest.main()
