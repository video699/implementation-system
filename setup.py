# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name='video699',
    packages=['video699'],
    version='v1.0.0a1+2018.04',
    description='System for lecture slide page retrieval based on lecture recordings',
    author='Vit Novotny',
    author_email='witiko@mail.muni.cz',
    url='https://pypi.org/project/video699-system/',
    project_urls={
        'Source': 'https://github.com/fi-muni-video-699/implementation-system',
        'Tracker': 'https://github.com/fi-muni-video-699/implementation-system/issues',
    },
    long_description=long_description,
    classifiers=[
        "Development Status :: 4 - Alpha",
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: Education",
        "Topic :: Multimedia :: Video :: Conversion",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
)
