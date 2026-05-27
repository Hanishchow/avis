"""Structured logging setup for the bird detection system."""

import os
import logging
import json
from datetime import datetime
from config import LOG_FILE


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        })


class ReadableFormatter(logging.Formatter):
    def format(self, record):
        return "%(asctime)s [%(levelname)s] %(message)s"


def setup_logger(name=None, json_output=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = JsonFormatter() if json_output else ReadableFormatter()

    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter if json_output else ReadableFormatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(sh)

    return logger
