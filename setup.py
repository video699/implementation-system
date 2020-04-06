# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst', 'r', encoding='utf8') as f:
    long_description = f.read()

setup(
    name='video699',
    packages=[
        'video699',
        'video699.document',
        'video699.event',
        'video699.frame',
        'video699.quadrangle',
        'video699.screen',
        'video699.video',
        # TODO Is this really right approach? Seems a little bit weird.
        'video699.screen.semantic_segmentation'
    ],
    package_data={
        'video699': [
            'configuration/default.ini',
        ],
        'video699.event': [
            'siamese/pretrained/classification_model.h5',
            'siamese/pretrained/training_moments.npz',
            'siamese/pretrained/training_history.pkl',
            'siamese/pretrained/format_version',
        ],
        'video699.screen': [
            'annotated/LICENSE',
            'annotated/*.xml',
            'annotated/*/*.xml',
        ],
        'video699.video': [
            'annotated/LICENSE',
            'annotated/*.xml',
            'annotated/*/*.xml',
            'annotated/*/*.png',
            'annotated/*/*.pdf',
        ],
    },
    version='1.0.0a1+2018.4',
    description='System for lecture slide page retrieval based on lecture recordings',
    author='Vit Novotny',
    author_email='witiko@mail.muni.cz',
    url='https://pypi.org/project/video699/',
    project_urls={
        'Source': 'https://github.com/video-699/implementation-system',
        'Tracker': 'https://github.com/video-699/implementation-system/issues',
    },
    install_requires=[
        'annoy~=1.13.0',
        'Keras~=2.2.4',
        'lxml~=4.2.4',
        'npstreams~=1.5.1',
        'numpy~=1.16.6',
        'opencv-python~=4.1.2',
        'PyMuPDF~=1.13.18',
        'python-dateutil~=2.7.3',
        'pyxdg~=0.26',
        'Rtree~=0.8.3',
        'scipy~=1.1.0',
        'Shapely~=1.6.4.post2',
        'tensorflow~=1.13.0',
        'fastai~=1.0.60',
        'scikit-learn~=0.22.2.post1',
    ],
    tests_require=[
        'codecov~=2.0.16',
        'coverage~=5.0.2',
    ],
    long_description=long_description,
    classifiers=[
        'Development Status :: 4 - Alpha',
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Education',
        'Topic :: Multimedia :: Video :: Conversion',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
)
