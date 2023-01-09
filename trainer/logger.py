import logging
from enum import IntEnum

class LogLevel(IntEnum):
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0

def create_logger(name:str, level: LogLevel, filename: str|None = None, no_stdout: bool = False) -> logging.Logger:    
    logger = logging.getLogger(name)
    logger.setLevel(int(level))
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                "%H:%M:%S")
    
    if not no_stdout:
        ch = logging.StreamHandler()
        ch.setLevel(int(level))
        ch.setFormatter(formatter)    
        logger.addHandler(ch)
    
    if filename is not None:
        fh = logging.FileHandler(filename=filename, encoding='utf-8')
        fh.setLevel(int(level))
        fh.setFormatter(formatter)    
        logger.addHandler(fh)
    
    return logger