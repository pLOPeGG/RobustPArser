import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from robust_parser import data


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a single-directional encoder so key_size is hidden_size
        key_size = hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.alphas = None

    def forward(self, decoder_hidden, encoder_hidden, mask=None):
        assert mask is not None, "mask is required"

        proj_key = self.key_layer(encoder_hidden)

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(decoder_hidden)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(-1)  # (InputSeqLen, Batch)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float("inf"))

        # Turn scores to probabilities.
        alphas = F.softmax(scores.rename(None), dim=0).unsqueeze(-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(
            alphas.transpose(0, 1).transpose(1, 2), encoder_hidden.transpose(0, 1)
        ).transpose(0, 1)

        # context shape: [1, B, D], alphas shape: [I, B]
        self.alphas = alphas.squeeze(-1)
        return context, self.alphas


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = BahdanauAttention(self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden, encoder_outputs, input_mask):
        embedded = self.embedding(x.rename(None))

        if isinstance(hidden, tuple):  # LSTM like rnn
            hidden = (hidden[0].rename(None), hidden[1].rename(None))
            hidden_h = hidden[0]
        else:
            hidden = hidden.rename(None)
            hidden_h = hidden

        attn_input, attn_probs = self.attn(hidden_h, encoder_outputs, input_mask)

        output, hidden = self.rnn(attn_input.rename(None), hidden)

        if isinstance(hidden, tuple):  # LSTM like rnn
            hidden = (
                hidden[0].refine_names(..., "B", "H"),
                hidden[1].refine_names(..., "B", "H"),
            )
        else:
            hidden = hidden.refine_names(..., "B", "H")

        output = output.refine_names("O", "B", "H")

        output = self.log_softmax(self.out(output))
        return output.refine_names(..., "V"), hidden


def visualize_attn(x, y, attn):
    batch_size = x.size(1)

    nrows, ncols = int(math.ceil(batch_size**.5)), int(math.ceil(batch_size**.5))
    fig, ax = plt.subplots(nrows, ncols)

    for batch_idx in range(batch_size):
        batch_tuple = divmod(batch_idx, ncols)
        src = [data.rev_vocabulary[i.item()] for i in x[:, batch_idx]]
        trg = [data.rev_vocabulary[i.item()] for i in y[:, batch_idx]]
        trg = trg[
            : next(
                i for i, v in enumerate(y[:, batch_idx]) if v.item() == data.vocabulary[data.__END__]
            )
        ]

        scores = np.transpose(attn[: len(trg), :, batch_idx], (1, 0))

        heatmap = ax[batch_tuple].pcolor(scores, cmap="viridis")

        ax[batch_tuple].set_xticklabels(trg, minor=False, rotation="vertical")
        ax[batch_tuple].set_yticklabels(src, minor=False)

        # put the major ticks at the middle of each cell
        # and the x-ticks on top
        ax[batch_tuple].xaxis.tick_top()
        ax[batch_tuple].set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
        ax[batch_tuple].set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
        ax[batch_tuple].invert_yaxis()

    for i in range(batch_size, nrows * ncols):
        fig.delaxes(ax.flatten()[i])
    # plt.colorbar(heatmap)
    plt.show()
