import pandas as pd
from abc import abstractmethod
from utils import get_logger, logging_conf
from model.metric import *

logger = get_logger(logger_conf=logging_conf)


class BaseTrainer:
    @abstractmethod
    def pretrain(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def valid(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def submission(self):
        raise NotImplementedError


# Make your class!
class TempTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__()

    def pretrain(self, args):
        pass

    def train(self, args):
        pass

    def validate(self, args):
        pass

    def test(self, args):
        results = 0
        return results

    def submission(self, args):
        result = pd.DataFrame()
        return result
