import logging
import logging.handlers
import os
import uuid
from pathlib import Path

KB = 1024
MB = 1024 * KB


def enable_dqo_logs():
    logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    logger = logging.getLogger('dqo')
    logger.setLevel(logging.INFO)


def rotating_logger(name: str = None, filename: str = None, flush_size: int = 10 * MB, max_files: int = 0, ctx: dict = None):
    basedir = ctx.get('basedir', '.') if ctx else '.'
    logger = logging.getLogger(name or str(uuid.uuid4()))
    filename = filename or name
    if filename is None:
        raise ValueError('either `filename` or `name` must be provided')

    path = os.path.join(basedir, filename)
    folder = os.path.dirname(path)
    Path(folder).mkdir(parents=True, exist_ok=True)

    ch = logging.handlers.RotatingFileHandler(path, 'a', flush_size, max_files)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.propagate = False

    return logger
