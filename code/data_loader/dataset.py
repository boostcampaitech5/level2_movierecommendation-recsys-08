from torch.utils.data import Dataset


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
