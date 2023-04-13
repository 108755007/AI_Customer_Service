import os, logging
from pathlib import Path
from datetime import datetime

class logger():
    def __init__(self, log_path='./log'):
        self.logger, self.handler = self.get_logger(log_path)

    def get_logger(self,log_path):
        date = datetime.today().strftime('%Y/%m/%d')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_file_path = os.path.join(log_path, '/'.join(date.split('/')[:-1]))
        Path(log_file_path).mkdir(parents=True, exist_ok=True)
        log_file_path = os.path.join(log_file_path, f"{date.split('/')[-1]}-1.log")
        while os.path.exists(log_file_path):
            log_file_path = '-'.join(log_file_path.split('-')[:-1] + [str(int(log_file_path.split('-')[-1][:-4]) + 1) + '.log'])
        with open(log_file_path, 'w+') as f:
            pass
        handler = logging.FileHandler(filename=log_file_path, mode='a')
        handler.setFormatter('%(asctime)s - %(filename)s[line:%(lineno)d] -\t %(levelname)s: %(message)s')
        handler.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO,
                            filename=log_file_path,
                            filemode='w',
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] -\t %(levelname)s: %(message)s')
        return logger, handler

    def print(self, *args, **kwargs):
        self.logger.setLevel(logging.DEBUG)
        level = kwargs.get('level', 'INFO')
        msg = ''.join(map(str, args))
        print(msg)
        if level == "DEBUG":
            logging.debug(msg)
        elif level == "WARNING":
            logging.warning(msg)
        elif level == "ERROR":
            logging.error(msg)
        else:
            logging.info(msg)
        self.logger.setLevel(logging.INFO)