import pandas as pd
from abc import abstractmethod
from utils.util import get_logger, logging_conf
from model.metric import *
from tqdm import tqdm

logger = get_logger(logger_conf=logging_conf)


class BaseTrainer:
    @abstractmethod
    def pretrain(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def valid(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def submission(self):
        raise NotImplementedError


# Make your class!
class TempTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__()

    def pretrain(self, args):
        pass

    def train(self, args):
        pass

    def validate(self, args):
        pass

    def test(self, args):
        results = 0
        return results

    def submission(self, args):
        result = pd.DataFrame()
        return result


class TransformerTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__()
        self.rating_matrix = dict()
        self.rating_matrix["valid"] = None
        self.rating_matrix["test"] = None
        self.rating_matrix["submission"] = None

    def get_data_iter(self, args, epoch, dataloader, mode="train"):
        if mode in ("train", "valid"):
            desc = desc = "%s-%d/%d" % (mode, epoch, args.epochs)
        else:
            desc = desc = "%s" % (mode)

        return tqdm(
            enumerate(dataloader),
            desc=desc,
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

    def predict_rating(self, item_emb, recommend_output):
        rating_pred = torch.matmul(recommend_output, item_emb.transpose(0, 1))
        return rating_pred

    def get_score(self, epoch, answers, pred_list, mode):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))

        results = dict()
        if mode == "valid":
            results["Epoch"] = epoch

        results["RECALL@5"] = "{:.4f}".format(recall[0])
        results["NDCG@5"] = "{:.4f}".format(ndcg[0])
        results["RECALL@10"] = "{:.4f}".format(recall[1])
        results["NDCG@10"] = "{:.4f}".format(ndcg[1])

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(results)

    def pretrain(self, args):
        pass

    def train(self, args):
        best_score = -1
        early_stopping_counter = 0
        for epoch in range(args.epochs):
            data_iter = self.get_data_iter(
                args, epoch, args.train_dataloader, mode="train"
            )

            args.model.train()
            avg_loss = 0.0
            cur_loss = 0.0

            for i, batch in data_iter:
                batch = tuple(t.to(args.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                seq_out = args.model(input_ids)

                loss = args.criterion(
                    args.model.item_embedding, seq_out, target_pos, target_neg
                )
                args.optimizer.zero_grad()
                loss.backward()
                args.optimizer.step()

                avg_loss += loss.item()
                cur_loss = loss.item()

            scores, results = self.validate(args, epoch)
            valid_score = scores[2]

            if valid_score > best_score:
                best_score = valid_score
                torch.save(
                    args.model.state_dict(), args.save_dir + args.saved_file_name
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    logger.info(
                        "EarlyStopping counter: %s out of %s",
                        early_stopping_counter,
                        args.patience,
                    )
                    break

            if (epoch + 1) % args.log_freq == 0:
                now = {
                    "epoch": epoch,
                    "avg_loss": "{:.4f}".format(avg_loss / len(data_iter)),
                    "cur_loss": "{:.4f}".format(cur_loss),
                }
                logger.info("LOSS : %s, SCORES : %s", str(now), results)

    def validate(self, args, epoch):
        return self.predict(
            args, epoch, args.valid_dataloader, args.valid_rating_matrix, mode="valid"
        )

    def test(self, args, epoch=0):
        _, results = self.predict(
            args, epoch, args.test_dataloader, args.test_rating_matrix, mode="test"
        )
        return results

    def submission(self, args, epoch=0):
        return self.predict(
            args,
            epoch,
            args.submission_dataloader,
            args.submission_rating_matrix,
            mode="submission",
        )

    def predict(self, args, epoch, dataloader, rating_matrix, mode="valid"):
        data_iter = self.get_data_iter(args, epoch, dataloader, mode=mode)

        args.model.eval()
        pred_list = None
        answer_list = None

        for i, batch in data_iter:
            batch = tuple(t.to(args.device) for t in batch)
            user_ids, input_ids, _, target_neg, answers = batch
            recommend_output = args.model(input_ids)
            recommend_output = recommend_output[:, -1, :]

            rating_pred = self.predict_rating(
                args.model.item_embedding.weight, recommend_output
            )
            rating_pred = rating_pred.cpu().data.numpy().copy()
            user_ids = user_ids.cpu().numpy()
            rating_pred[rating_matrix[user_ids].toarray() > 0] = 0

            ind = np.argpartition(rating_pred, -10)[:, -10:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, np.newaxis], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[
                np.arange(len(rating_pred))[:, np.newaxis], arr_ind_argsort
            ]

            if i == 0:
                pred_list = batch_pred_list
                answer_list = answers.cpu().data.numpy()
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

        if mode == "submission":
            result = list()

            for idx, items in enumerate(pred_list):
                for item in items:
                    result.append((args.users[idx], item))

            return result
        else:
            return self.get_score(epoch, answer_list, pred_list, mode)
