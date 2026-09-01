"""Structured application logging."""
import logging, sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger