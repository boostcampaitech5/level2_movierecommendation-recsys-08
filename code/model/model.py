import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import copy
from math import sqrt


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


class SASRec(BaseModel):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.device = args.device
        self.item_embedding = nn.Embedding(args.num_items, args.hidden_size)
        self.position_embedding = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12).to(args.device)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layers = nn.ModuleList(
            [copy.deepcopy(Block(args)) for _ in range(args.num_hidden_layers)]
        )

        self.initializer_range = args.initializer_range
        if not args.using_pretrain:
            self.apply(self.init_weights)

    def make_subsequent_mask(self, x, pad_idx=0):
        max_seq_length = x.size(-1)

        attention_shape = (1, max_seq_length, max_seq_length)
        subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == pad_idx).unsqueeze(1)
        subsequent_mask.requires_grad = False

        return subsequent_mask

    def make_pad_mask(self, x, pad_idx=0):
        max_seq_length = x.size(-1)

        row_wise = x.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        row_wise = row_wise.repeat(1, 1, 1, max_seq_length)

        column_wise = x.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        column_wise = column_wise.repeat(1, 1, max_seq_length, 1)

        pad_mask = row_wise & column_wise
        pad_mask.requires_grad = False

        return pad_mask

    def add_position_embedding(self, x):
        max_seq_length = x.size(-1)
        item_embedding = self.item_embedding(x)

        position_ids = torch.arange(max_seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        position_embedding = self.position_embedding(position_ids)

        sequence_emb = item_embedding + position_embedding
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        out = self.add_position_embedding(x)

        pad_mask = self.make_pad_mask(x).to(self.device)
        subsequent_mask = self.make_subsequent_mask(x).to(self.device)
        mask = pad_mask & subsequent_mask

        for layer in self.layers:
            out = layer(out, mask)

        return out


class Block(nn.Module):
    def __init__(self, args):
        super(Block, self).__init__()
        self.self_attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)
        self.add_norms = [Add_Norm(args) for _ in range(2)]

    def forward(self, x, mask=None):
        out = self.add_norms[0](x, lambda x: self.self_attention(x, mask))
        out = self.add_norms[1](out, self.feed_forward)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.model_size = args.hidden_size * args.num_attention_heads
        self.n_heads = args.num_attention_heads
        self.q_fc = nn.Linear(args.hidden_size, self.model_size)
        self.k_fc = nn.Linear(args.hidden_size, self.model_size)
        self.v_fc = nn.Linear(args.hidden_size, self.model_size)
        self.out_fc = nn.Linear(self.model_size, args.hidden_size)

    def forward(self, x, mask=None):
        n_batch = x.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.n_heads, self.model_size // self.n_heads)
            out = out.transpose(1, 2)
            return out

        Q = transform(x, self.q_fc)
        K = transform(x, self.k_fc)
        V = transform(x, self.v_fc)

        def calculate_attention(Q, K, V, mask):
            d_k = K.shape[-1]
            attention_score = torch.matmul(Q, K.transpose(-2, -1))
            attention_score = attention_score / sqrt(d_k)
            if mask is not None:
                attention_score = attention_score.masked_fill(mask == 0, -1e12)
            attention_score = torch.softmax(attention_score, dim=-1)
            out = torch.matmul(attention_score, V)
            return out

        out = calculate_attention(Q, K, V, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n_batch, -1, self.model_size)
        out = self.out_fc(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(args.hidden_size * 4, args.hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out


class Add_Norm(nn.Module):
    def __init__(self, args):
        super(Add_Norm, self).__init__()
        self.norm = nn.LayerNorm(args.hidden_size).to(args.device)
        self.drop_out = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, x, layer):
        out = x
        out = layer(out)
        out = self.norm(self.drop_out(out) + x)
        return out
