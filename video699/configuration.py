# -*- coding: utf-8 -*-

"""This module loads the default configuration and provides it to the rest of the package.

"""

from configparser import ConfigParser
import os


DEFAULT_CONFIGURATION_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'configuration',
    'default.ini',
)
CONFIGURATION = ConfigParser()


with open(DEFAULT_CONFIGURATION_PATHNAME) as f:
    CONFIGURATION.read_file(f)
