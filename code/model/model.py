import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self):
        raise NotImplementedError


# Make your class!
class TempModel(BaseModel):
    def __init__(self, args):
        super().__init__()

    def forward(self):
        pass
