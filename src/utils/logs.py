import sys
import logging
from typing import Text, Union


def get_logger(name: Text, log_level: Union[Text, int]):
    """Get logger.
    Args:
        name {Text}: logger name
        log_level {Text or int}: logging level; can be string or integer value
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    streamHandler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    streamHandler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(streamHandler)
    logger.propagate = False

    return logger