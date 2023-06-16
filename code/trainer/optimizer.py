import torch
from torch.optim import Adam, AdamW


def get_optimizer(args):
    if args.optimizer_title == "Adam":
        optimizer = Adam(
            args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    elif args.optimizer_title == "AdamW":
        optimizer = AdamW(
            args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    return optimizer
