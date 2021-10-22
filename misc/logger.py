import logging
import os
import time
from datetime import datetime
from .F import ensure_folder

logging_DEBUG = logging.DEBUG
logging_INFO = logging.INFO
logging_WARNING = logging.WARNING
logging_ERROR = logging.ERROR
logging_CRITICAL = logging.CRITICAL


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    starttime = time.time()
    current_time = datetime.now().strftime('%Y%m%d_%H%M')

    def __init__(self, logdir=os.getcwd(), dirname='log_XD', timestamp=current_time, level=logging_INFO):
        self._timestamp = timestamp
        self._path = os.path.join(os.path.join(logdir, dirname), self._timestamp)
        ensure_folder(self._path, remake=True)
        logging.basicConfig(filename=os.path.join(self._path, 'log.txt'), level=level)  # 僅第一次調用有效果

    def get_log_dir(self):
        return self._path

    @staticmethod
    def info(m):
        logging.info(m)

    @staticmethod
    def debug(m):
        logging.debug(m)

    @staticmethod
    def warning(m):
        logging.warning(m)

    @staticmethod
    def error(m):
        logging.error(m)


if __name__ == "__main__":
    logger = Logger(level=logging_INFO)
    logger.info(id(logger))
    logger.info(id(Logger(level=logging_INFO)))

