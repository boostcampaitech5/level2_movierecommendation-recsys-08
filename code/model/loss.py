import torch
from torch.nn import BCELoss


def get_loss(args):
    if args.loss_title == "BCELoss":
        criterion = BCELoss()

    return criterion
