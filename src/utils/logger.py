import logging
import sys

LOG_FORMAT = "%(levelname)s:   %(message)s"


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
