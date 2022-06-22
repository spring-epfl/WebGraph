"""Logging utility"""

import logging

LOGGER_NAME = "webgraph"
LOGGER_FORMAT = "%(levelname)s %(message)s"
LOGGER_LEVEL = logging.WARNING


def configure_logger() -> logging.Logger:
    """Configure the logger used by Webgraph."""
    formatter = logging.Formatter(LOGGER_FORMAT)

    handler = logging.StreamHandler()
    handler.setLevel(LOGGER_LEVEL)
    handler.setFormatter(formatter)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LOGGER_LEVEL)
    logger.addHandler(handler)

    return logger


LOGGER = configure_logger()
