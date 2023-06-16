import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import data_loader.data_preprocessor as module_data_preprocessor
import data_loader.dataset as module_dataset
import model.model as module_model
import trainer.trainer as module_trainer
from utils import parse_args, set_seeds, get_logger, logging_conf, checkDirectory

# import wandb

logger = get_logger(logging_conf)


def main(args):
    # wandb.login()
    # wandb.init(project="movieRecommendation", config=vars(args))

    set_seeds(args.seed)
    checkDirectory(args.submission_dir)

    logger.info("Preparing data ...")
    args.preprocessor = getattr(module_data_preprocessor, args.data_preprocessor_title)(
        args
    )
    args.preprocessor.preprocessing(args)

    submission_dataset = getattr(module_dataset, args.dataset_title)(
        args, data_type="submission"
    )
    args.submission_dataloader = DataLoader(
        submission_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    logger.info("Building Model ...")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = getattr(module_model, args.model_title)(args).to(args.device)
    args.model.load_state_dict(torch.load(args.save_dir + args.saved_file_name))

    logger.info("Building Submission ...")
    trainer = getattr(module_trainer, args.trainer_title)(args)
    result = trainer.submission(args)

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        args.submission_dir + args.submission_file_name, index=False
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
