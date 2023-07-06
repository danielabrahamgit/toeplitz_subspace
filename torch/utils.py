import logging

def setup_console_logger(
        logger,
        log_level: int,
        fmt: str = '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handler_cls = logging.StreamHandler,
):
    logger.setLevel(log_level)
    formatter = logging.Formatter(fmt)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)
            return
    handler = handler_cls()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
