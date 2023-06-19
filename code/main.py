import os
import torch
from torch.utils.data import DataLoader
import data_loader.data_preprocessor as module_data_preprocessor
import data_loader.dataset as module_dataset
import model.model as module_model
from model.loss import get_loss
from trainer.optimizer import get_optimizer
from trainer.scheduler import get_scheduler
import trainer.trainer as module_trainer
from utils.util import parse_args, set_seeds, get_logger, logging_conf, checkDirectory

# import wandb

logger = get_logger(logging_conf)


def main(args):
    # wandb.login()
    # wandb.init(project="movieRecommendation", config=vars(args))

    set_seeds(args.seed)
    checkDirectory(args.save_dir)

    logger.info("Preparing data ...")
    args.preprocessor = getattr(module_data_preprocessor, args.data_preprocessor_title)(
        args
    )
    args.preprocessor.preprocessing(args)

    train_dataset = getattr(module_dataset, args.dataset_title)(args, data_type="train")
    args.train_dataloader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    valid_dataset = getattr(module_dataset, args.dataset_title)(args, data_type="valid")
    args.valid_dataloader = DataLoader(
        valid_dataset,
        num_workers=args.num_workers,
        shuffle=False,
        batch_size=args.batch_size,
    )

    test_dataset = getattr(module_dataset, args.dataset_title)(args, data_type="test")
    args.test_dataloader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        shuffle=False,
        batch_size=args.batch_size,
    )

    logger.info("Building Model ...")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = getattr(module_model, args.model_title)(args).to(args.device)

    logger.info("Start Training ...")
    args.criterion = get_loss(args)
    args.optimizer = get_optimizer(args)
    args.scheduler = get_scheduler(args)

    trainer = getattr(module_trainer, args.trainer_title)(args)

    if args.using_pretrain:
        checkDirectory(args.pretrain_dir)
        pretrained_path = os.path.join(args.pretrain_dir, args.pretrain_file_name)

        try:
            print(f"Load Checkpoint From {pretrained_path}!")
            args.model.load_state_dict(torch.load(pretrained_path))

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found!")

            pretrain_dataset = getattr(module_dataset, args.dataset_title)(
                args, data_type="pretrain"
            )
            args.pretrain_dataloader = DataLoader(
                pretrain_dataset,
                num_workers=args.num_workers,
                shuffle=True,
                batch_size=args.batch_size,
            )

            trainer.pretrain(args)
            args.model.load_state_dict(torch.load(pretrained_path))
    else:
        print("The Model is not pretrained.")

    trainer.train(args)

    logger.info("Model Results ...")
    args.model.load_state_dict(torch.load(args.save_dir + args.saved_file_name))
    results = trainer.test(args)
    print(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
