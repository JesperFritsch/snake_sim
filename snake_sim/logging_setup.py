import logging

from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(log_level, log_dir=None):
    default_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(default_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    log_file = Path(log_dir) / 'snake.log' if log_dir else Path('snake.log').absolute()
    log_file.parent.mkdir(parents=True, exist_ok=True)


    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=2, delay=True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(default_formatter)
    root_logger.addHandler(file_handler)