import logging

class LoginLogger:
    def __init__(self, log_file='login_events.log'):
        self.logger = logging.getLogger('LoginLogger')
        self.logger.setLevel(logging.INFO)
        self.log_file = log_file
        self._setup_handler()

    def _setup_handler(self):
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def log(self, message):
        self.logger.info(message)

# def setup_login_logger(log_file='login_events.log'):
#     logger = logging.getLogger('LoginLogger')
#     logger.setLevel(logging.INFO)

#     file_handler = logging.FileHandler(log_file)
#     formatter = logging.Formatter('%(asctime)s - %(message)s')
#     file_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)
#     return logger

# login_logger = setup_login_logger()

# __all__ = ['login_logger']
