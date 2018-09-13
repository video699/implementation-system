# -*- coding: utf-8 -*-

import os
import unittest

import cv2 as cv

from video699.document.pdf import PDFDocument


DOCUMENT_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'test_pdf',
    'sample_pdf_document.pdf',
)
DOCUMENT_TITLE = 'Example title'
DOCUMENT_AUTHOR = 'Example author'
PAGE_IMAGE_WIDTH = 2550
PAGE_IMAGE_HEIGHT = 3300
PAGE_IMAGE_HORIZONTAL_MARGIN = 100
PAGE_IMAGE_VERTICAL_MARGIN = 200


class TestPDFDocument(unittest.TestCase):
    """Tests the ability of the PDFDocument class to read a PDF document file.

    """

    def setUp(self):
        self.document = PDFDocument(DOCUMENT_PATHNAME)

    def test_document_properties(self):
        self.assertEqual(DOCUMENT_TITLE, self.document.title)
        self.assertEqual(DOCUMENT_AUTHOR, self.document.author)

    def test_reads_two_pages(self):
        page_iterator = iter(self.document)
        next(page_iterator)
        next(page_iterator)
        with self.assertRaises(StopIteration):
            next(page_iterator)


class TestPDFDocumentPage(unittest.TestCase):
    """Tests the ability of the PDFDocumentPage class to render pages and produce page numbers.

    """

    def setUp(self):
        self.document = PDFDocument(DOCUMENT_PATHNAME)
        page_iterator = iter(self.document)
        self.first_page = next(page_iterator)
        self.second_page = next(page_iterator)
        self.first_page_image = self.first_page.image(PAGE_IMAGE_WIDTH, PAGE_IMAGE_HEIGHT)
        self.second_page_image = self.second_page.image(PAGE_IMAGE_WIDTH, PAGE_IMAGE_HEIGHT)
        self.page_image_with_wider_aspect_ratio = self.first_page.image(
            PAGE_IMAGE_WIDTH + PAGE_IMAGE_HORIZONTAL_MARGIN,
            PAGE_IMAGE_HEIGHT,
        )
        self.page_image_with_taller_aspect_ratio = self.first_page.image(
            PAGE_IMAGE_WIDTH,
            PAGE_IMAGE_HEIGHT + PAGE_IMAGE_VERTICAL_MARGIN,
        )

    def test_document(self):
        self.assertEqual(self.document, self.first_page.document)
        self.assertEqual(self.document, self.second_page.document)

    def test_page_number(self):
        self.assertEqual(1, self.first_page.number)
        self.assertEqual(2, self.second_page.number)

    def test_first_page_image(self):
        image = self.first_page_image
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
        image = self.second_page_image
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
        image = self.page_image_with_wider_aspect_ratio
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
        image = self.page_image_with_taller_aspect_ratio
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
