from torch.utils.data import Dataset
from utils.util import negative_sampling
import torch


# Make your class!
class TempDataset(Dataset):
    def __init__(self, args, data_type="train"):
        super().__init__()
        self.data_type = data_type
        if self.data_type == "pretrain":
            pass
        elif self.data_type == "train":
            pass
        elif self.data_type == "valid":
            pass
        elif self.data_type == "test":
            pass
        else:  # submission
            pass

    def __getitem__(self, index):
        if self.data_type == "pretrain":
            pass
        elif self.data_type == "train":
            pass
        elif self.data_type == "valid":
            pass
        elif self.data_type == "test":
            pass
        else:  # submission
            pass

    def __len__(self):
        pass


class TransformerDataset(Dataset):
    def __init__(self, args, data_type="train"):
        super().__init__()
        self.data_type = data_type
        self.augmentation = True
        self.train_sequences = args.train_sequences
        self.user_sequences = args.user_sequences
        self.all_items = args.all_items
        self.num_items = args.num_items
        self.max_seq_length = args.max_seq_length

        assert self.data_type in {"train", "valid", "test", "submission"}

    def __getitem__(self, index):
        if self.augmentation:
            if self.data_type == "train":
                items = self.train_sequences[index]
            else:
                items = self.user_sequences[index]
        else:
            items = self.user_sequences[index]

        if self.data_type == "pretrain":
            pass

        elif self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [items[-3]]  # will not be used

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        else:  # submission
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []  # will not be used

        target_neg = []
        for _ in target_pos:
            target_neg.append(negative_sampling(set(items), self.all_items))

        # padding
        pad_len = self.max_seq_length - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # truncation
        input_ids = input_ids[-self.max_seq_length :]
        target_pos = target_pos[-self.max_seq_length :]
        target_neg = target_neg[-self.max_seq_length :]

        assert len(input_ids) == self.max_seq_length
        assert len(target_pos) == self.max_seq_length
        assert len(target_neg) == self.max_seq_length

        return (
            torch.tensor(index, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

    def __len__(self):
        if self.augmentation:
            if self.data_type == "train":
                return len(self.train_sequences)
            else:
                return len(self.user_sequences)
        else:
            return len(self.user_sequences)
