# -*- coding: utf-8 -*-

"""This module loads the package configuration files from standard XDG directories.

"""

from configparser import ConfigParser
from logging import getLogger
import os

from xdg.BaseDirectory import load_config_paths


CONFIGURATION = ConfigParser()
RESOURCE_NAME = 'video699'
DEFAULT_CONFIGURATION_FILE_PATHNAME = os.path.join(
    os.path.dirname(__file__),
    'configuration',
    'default.ini',
)
WORKING_DIRECTORY_CONFIGURATION_FILE_PATHNAME = os.path.join(
    '.',
    '{}.ini'.format(RESOURCE_NAME),
)
CONFIGURATION_FILE_PATHNAMES = [
    WORKING_DIRECTORY_CONFIGURATION_FILE_PATHNAME,
    *reversed(list(load_config_paths(RESOURCE_NAME))),
    DEFAULT_CONFIGURATION_FILE_PATHNAME,
]
LOGGER = getLogger(__name__)


def get_configuration():
    """Returns the package configuration.

    Returns
    -------
    configuration : ConfigParser
        The package configuration.
    """
    return CONFIGURATION


for pathname in CONFIGURATION_FILE_PATHNAMES:
    LOGGER.debug("Reading configuration file {}".format(pathname))
    try:
        with open(pathname) as f:
            CONFIGURATION.read_file(f)
    except OSError as err:
        LOGGER.debug("Failed to read configuration file {0}: {1}".format(pathname, err))
