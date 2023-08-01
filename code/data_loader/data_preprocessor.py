from abc import abstractmethod
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


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


class TransformerDataPreprocessor(BaseDataPreprocessor):
    def __init__(self, args):
        super().__init__()
        args.data_file = args.data_dir + "train_ratings.csv"

    def preprocessing(self, args):
        self.get_user_sequences(args)
        self.get_all_items(args)
        self.augment_data(args)
        args.valid_rating_matrix = self.generate_rating_matrix(args, "valid")
        args.test_rating_matrix = self.generate_rating_matrix(args, "test")
        args.submission_rating_matrix = self.generate_rating_matrix(args, "submission")

    def get_user_sequences(self, args):
        df = pd.read_csv(args.data_file)
        df = df.sort_values(by=["user", "time"], axis=0)

        args.user_sequences = df.groupby("user")["item"].apply(list).to_list()
        args.users = df["user"].unique()
        args.num_users = len(args.user_sequences)
        args.num_items = df["item"].max() + 2  # padding, masking(pretrain)

    def get_all_items(self, args):
        df = pd.read_csv(args.data_file)
        args.all_items = df["item"].unique()

    def augment_data(self, args):
        args.train_sequences = list()
        length = args.max_seq_length + 3
        sliding = 2
        limit = 5
        for user_sequence in args.user_sequences:
            n = len(user_sequence)
            if n > length:
                now = -length
                threshold = -n
                k = 0
                while now >= threshold and k < limit:
                    if now + length == 0:
                        args.train_sequences.append(user_sequence[now:])
                    else:
                        args.train_sequences.append(user_sequence[now : now + length])
                    now -= sliding
                    k += 1
            else:
                args.train_sequences.append(user_sequence)

    def generate_rating_matrix(self, args, mode="valid"):
        row = []
        col = []
        data = []

        for user_id, item_list in enumerate(args.user_sequences):
            if mode == "valid":
                item_list = item_list[:-2]

            elif mode == "test":
                item_list = item_list[:-1]

            else:
                item_list = item_list[:]

            for item in item_list:
                row.append(user_id)
                col.append(item)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)

        rating_matrix = csr_matrix(
            (data, (row, col)), shape=(args.num_users, args.num_items)
        )

        return rating_matrix
