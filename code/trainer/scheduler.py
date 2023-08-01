import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_scheduler(args):
    if args.scheduler_title == "plateau":
        scheduler = ReduceLROnPlateau(
            args.optimizer,
            patience=args.patience,
            factor=args.factor,
            mode="max",
            verbose=True,
        )

    return scheduler
