# -*- coding: utf-8 -*-

"""This module loads the package configuration from standard XDG directories.

"""

from configparser import ConfigParser
from logging import getLogger
import os

from xdg.BaseDirectory import load_config_paths


CONFIGURATION = ConfigParser()
DEFAULT_CONFIGURATION_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'configuration',
    'default.ini',
)
LOGGER = getLogger(__name__)
RESOURCE_NAME = 'video699'


def get_configuration():
    """Returns the package configuration.

    Returns
    -------
    configuration : ConfigParser
        The package configuration.
    """
    return CONFIGURATION


for pathname in reversed(list(load_config_paths(RESOURCE_NAME)) + [DEFAULT_CONFIGURATION_PATHNAME]):
    LOGGER.debug("Reading configuration file {}".format(pathname))
    with open(pathname) as f:
        CONFIGURATION.read_file(f)
