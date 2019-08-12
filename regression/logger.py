import time

import numpy as np

import os
import sys
import logging
import pathlib
from datetime import datetime
from pytz import timezone, utc
from log_helper import LogHelper

class Logger:

    def __init__(self, model_type):
        self.train_loss = []
        self.train_conf = []

        self.valid_loss = []
        self.valid_conf = []

        self.test_loss = []
        self.test_conf = []

        self.best_valid_model = None

        output_dir = './artifacts/train/{}_{}'.format(model_type, datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                        log_level=logging.INFO)
        self._logger = logging.getLogger(__name__)

    def print_info(self, iter_idx, start_time):


        self._logger.info(
            'Iter {:<4} - time: {:<5} - [train] loss: {:<6} (+/-{:<6}) - [valid] loss: {:<6} (+/-{:<6}) - [test] loss: {:<6} (+/-{:<6})'.format(
                iter_idx,
                int(time.time() - start_time),
                np.round(self.train_loss[-1], 4),
                np.round(self.train_conf[-1], 4),
                np.round(self.valid_loss[-1], 4),
                np.round(self.valid_conf[-1], 4),
                np.round(self.test_loss[-1], 4),
                np.round(self.test_conf[-1], 4),
            )
        )
