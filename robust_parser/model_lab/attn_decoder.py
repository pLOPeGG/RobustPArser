import torch


from torch import nn
from torch.nn import functional as F


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
