import torch
from torch.nn import BCELoss


class TransformerLoss(torch.nn.Module):
    def __init__(self, args):
        super(TransformerLoss, self).__init__()
        self.item_embedding = args.model.item_embedding
        self.hidden_size = args.hidden_size
        self.max_seq_length = args.max_seq_length

    def forward(self, item_embedding, seq_out, pos_ids, neg_ids):
        pos_emb = item_embedding(pos_ids)
        neg_emb = item_embedding(neg_ids)

        pos = pos_emb.view(-1, self.hidden_size)
        neg = neg_emb.view(-1, self.hidden_size)
        seq_emb = seq_out.view(-1, self.hidden_size)

        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.max_seq_length).float()
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss


def get_loss(args):
    if args.loss_title == "BCELoss":
        criterion = BCELoss()

    elif args.loss_title == "TransformerLoss":
        criterion = TransformerLoss(args)

    return criterion
