import logging
import os


def get_logger(log_dir="logs", name="experiment"):

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log")
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger