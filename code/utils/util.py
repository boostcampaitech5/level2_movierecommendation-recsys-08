import os
import random
import numpy as np
import torch
import argparse


def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def checkDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def negative_sampling(item_set, all_items):
    item = random.choice(all_items)
    while item in item_set:
        item = random.choice(all_items)
    return item


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=32, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--pretrain_dir", default="./pretrain_models/", type=str)
    parser.add_argument("--pretrain_file_name", default="pretrain.pt", type=str)
    parser.add_argument("--save_dir", default="./saved_models/", type=str)
    parser.add_argument("--saved_file_name", default="best_model.pt", type=str)
    parser.add_argument("--submission_dir", default="./submit/", type=str)
    parser.add_argument("--submission_file_name", default="submission.csv", type=str)

    # model args
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.25,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=400, type=int)

    # train args
    parser.add_argument("--num_workers", type=int, default=2, help="num_workers")
    parser.add_argument(
        "--lr", type=float, default=0.00005, help="learning rate of adam"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.7, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--patience", type=int, default=10, help="patience")
    parser.add_argument("--factor", default=0.5, type=float, help="optimizer factor")

    # pre train args
    parser.add_argument(
        "--pre_epochs", type=int, default=300, help="number of pre_train epochs"
    )
    parser.add_argument("--pre_batch_size", type=int, default=512)
    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
    parser.add_argument("--mip_weight", type=float, default=1.0, help="mip loss weight")
    parser.add_argument("--map_weight", type=float, default=1.0, help="map loss weight")
    parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

    # User-Specific Settings
    parser.add_argument(
        "--using_pretrain", type=bool, default=True, help="use to pretrain"
    )
    parser.add_argument(
        "--data_preprocessor_title",
        default="TransformerDataPreprocessor",
        type=str,
        help="data_preprocessor",
    )
    parser.add_argument(
        "--dataset_title", default="TransformerDataset", type=str, help="dataset"
    )
    parser.add_argument("--model_title", default="SASRec", type=str, help="model")
    parser.add_argument(
        "--loss_title", default="TransformerLoss", type=str, help="loss"
    )
    parser.add_argument(
        "--optimizer_title", default="Adam_betas", type=str, help="optimizer"
    )
    parser.add_argument(
        "--scheduler_title", default="plateau", type=str, help="scheduler"
    )
    parser.add_argument(
        "--trainer_title", default="TransformerTrainer", type=str, help="trainer"
    )

    args = parser.parse_args()

    return args


def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
