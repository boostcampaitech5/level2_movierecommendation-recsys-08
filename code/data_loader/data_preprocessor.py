from abc import abstractmethod


class BaseDataPreprocessor:
    @abstractmethod
    def preprocessing(self):
        raise NotImplementedError

    @abstractmethod
    def load_data_from_file(self):
        raise NotImplementedError


# Make your class!
class TempDataPreprocessor(BaseDataPreprocessor):
    def __init__(self, args):
        super().__init__()

    def preprocessing(self, args):
        pass
