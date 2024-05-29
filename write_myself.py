import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import mypackage as mp
from torchsummary import summary

# batch_size = 128
# seqlen = 64
# dim = 512
# d_model = 512
# n_head = 8

# X = torch.randn(batch_size, seqlen, dim)
# print(X.shape)


# Multi-head attention
class multi_head_attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(multi_head_attention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_combine = nn.Linear(self.d_model, self.d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, seqlen, dim = q.shape
        d_head = self.d_model // self.n_head  # 每个head的维度
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch_size, seqlen, self.n_head, d_head).permute(0, 2, 1, 3)
        k = k.view(batch_size, seqlen, self.n_head, d_head).permute(0, 2, 1, 3)
        v = v.view(batch_size, seqlen, self.n_head, d_head).permute(0, 2, 1, 3)

        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_head)  # 这里的除法是广播的，对每个元素都除
        if mask is not None:
            # mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool))
            # mask = torch.tril(torch.ones(seqlen, seqlen))
            score = score.masked_fill(mask == 0, -1e6)
        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch_size, seqlen, self.d_model)
        output = self.w_combine(score)
        return output

# attention = multi_head_attention(d_model, n_head)
# out = attention(X, X, X)
# print(out, out.shape)


# Token Embedding
class token_embedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(token_embedding, self).__init__(vocab_size, d_model, padding_idx=1)


# Position Embedding
class position_embedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(position_embedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model)) # 上面的unsqueeze(1)是为了广播,在除法的时候进行了广播
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]


# Total Embedding
class total_embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(total_embedding, self).__init__()
        self.token_embedding = token_embedding(vocab_size, d_model)
        self.position_embedding = position_embedding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(x)
        return self.dropout(token_embed + pos_embed)  # size (seqlen, d_model),maybe batch?


# Layer Normalization
class layer_norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(layer_norm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # layer归一化的时候是在最后一个维度上进行的
        std = x.std(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(std + self.eps)  # # -和/都做了广播
        out = self.gamma * out + self.beta  # gamma和beta也是广播的
        return out


# Position-wise Feed-Forward
class position_wise_feed_forward(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob):
        super(position_wise_feed_forward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_prob)  # Dropout和Linear一样，都是在最后一个维度上进行的

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Encoder Layer
class encoder_layer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob):
        super(encoder_layer, self).__init__()
        self.attention = multi_head_attention(d_model, n_head)
        self.lnorm1 = layer_norm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = position_wise_feed_forward(d_model, d_ff, drop_prob)
        self.lnorm2 = layer_norm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.lnorm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.lnorm2(x + _x)

        return x


# Decoder Layer
class decoder_layer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob):
        super(decoder_layer, self).__init__()
        self.self_attention = multi_head_attention(d_model, n_head)
        self.lnorm1 = layer_norm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = multi_head_attention(d_model, n_head)
        self.lnorm2 = layer_norm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = position_wise_feed_forward(d_model, d_ff, drop_prob)
        self.lnorm3 = layer_norm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, x, enc, seq_mask, pad_mask):
        _x = x
        x = self.self_attention(x, x, x, seq_mask)  # 下三角掩码，因果掩码，挡住未来信息
        x = self.dropout1(x)
        x = self.lnorm1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, pad_mask)  # padding的掩码
            x = self.dropout2(x)
            x = self.lnorm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.lnorm3(x + _x)

        return x


# Encoder
class encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_len, d_model, n_head, d_ff, n_layer, device, drop_prob):
        super(encoder, self).__init__()

        self.embedding = total_embedding(enc_vocab_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [encoder_layer(d_model, n_head, d_ff, drop_prob) for _ in range(n_layer)]
        )

    def forward(self, x, pad_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x


# Decoder
class decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_len, d_model, n_head, d_ff, n_layer, device, drop_prob):
        super(decoder, self).__init__()

        self.embedding = total_embedding(dec_vocab_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [decoder_layer(d_model, n_head, d_ff, drop_prob) for _ in range(n_layer)]
        )

        self.fc = nn.Linear(d_model, dec_vocab_size)

    def forward(self, x, enc, seq_mask, pad_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc, seq_mask, pad_mask)

        x = self.fc(x)
        return x


# Transformer
class transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_vocab_size, dec_vocab_size,
                 max_len, d_model, n_head, d_ff, n_layer, device, drop_prob):
        super(transformer, self).__init__()

        self.encoder = encoder(enc_vocab_size, max_len, d_model, n_head, d_ff, n_layer, device, drop_prob)
        self.decoder = decoder(dec_vocab_size, max_len, d_model, n_head, d_ff, n_layer, device, drop_prob)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 下三角掩码，因果掩码
    def make_seq_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    # padding掩码
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)

        # (batch_size, seq_len, len_q, len_k)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k  # 逻辑与, 两个都是True才是True, 也就是两个都不是pad_idx才是True, 也就是有一个是pad_idx就被mask掉
        return mask

    def forward(self, src, trg):
        src_pad_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)  # for encoder
        trg_pad_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)  # for decoder
        trg_seq_mask = self.make_seq_mask(trg, trg)
        trg_mask = trg_pad_mask * trg_seq_mask
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_pad_mask)
        out = self.decoder(trg, enc, trg_mask, src_trg_mask)

        return out


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


if __name__ == "__main__":
    enc_vocab_size = 5893
    dec_vocab_size = 7853
    src_pad_idx = 1
    trg_pad_idx = 1
    # trg_sos_idx = 2
    batch_size = 128
    max_len = 1024
    d_model = 512
    n_layers = 3
    n_heads = 2
    ffn_hidden = 1024
    drop_prob = 0.1
    device = mp.try_cuda()

    model = transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        enc_vocab_size=enc_vocab_size,
        dec_vocab_size=dec_vocab_size,
        max_len=max_len,
        d_model=d_model,
        n_head=n_heads,
        d_ff=ffn_hidden,
        n_layer=n_layers,
        device=device,
        drop_prob=drop_prob
    ).to(device)

    model.apply(initialize_weights)
    src = torch.load('D:\dl_code\jibengong\Transformer\\tensor_src.pt')
    src = torch.cat((src, torch.ones(src.shape[0], 2, dtype=torch.int)), dim=-1)
    trg = torch.load('D:\dl_code\jibengong\Transformer\\tensor_trg.pt')
    print(src[1], src.shape)
    print(trg[1], trg.shape)

    result = model(src, trg)
    summary(model, src, trg)
    print(result.shape)
