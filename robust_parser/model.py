import torch

from torch import nn
from torch.nn import functional as F

from robust_parser import config, data


class MogrifierRNN(nn.Module):
    def __init__(self, rnn_cell, n_mogrifying):
        super().__init__()

        self.cell = rnn_cell
        if isinstance(self.cell, nn.LSTMCell):
            self.deal_tuple_hidden = True
        else:
            self.deal_tuple_hidden = False

        self.n_mogrifying = n_mogrifying

        self.m, self.n = rnn_cell.input_size, rnn_cell.hidden_size

        self.k = min(self.m, self.n) // 2

        # fmt: off
        self.Q_left = nn.ParameterList([nn.Parameter(torch.empty((self.m, self.k), dtype=torch.float, requires_grad=True)) for _ in range(1, n_mogrifying, 2)])
        self.Q_right = nn.ParameterList([nn.Parameter(torch.empty((self.k, self.n), dtype=torch.float, requires_grad=True)) for _ in range(1, n_mogrifying, 2)])
        self.R_left = nn.ParameterList([nn.Parameter(torch.empty((self.n, self.k), dtype=torch.float, requires_grad=True)) for _ in range(2, n_mogrifying, 2)])
        self.R_right = nn.ParameterList([nn.Parameter(torch.empty((self.k, self.m), dtype=torch.float, requires_grad=True)) for _ in range(2, n_mogrifying, 2)])
        # fmt: on

        self.init_weights()

    def init_weights(self):
        for w_l in [self.Q_left, self.Q_right, self.R_left, self.R_right]:
            for w in w_l:
                nn.init.xavier_uniform_(w)

    def mogrify(self, xx, hidden):
        if self.deal_tuple_hidden:
            h = hidden[0]
        else:
            h = hidden

        r_iter = zip(self.R_left, self.R_right)
        for q_l, q_r in zip(self.Q_left, self.Q_right):
            q = q_l @ q_r
            xx = 2 * torch.sigmoid(torch.matmul(q, h.unsqueeze(-1)).squeeze(-1)) * xx

            try:
                r_l, r_r = next(r_iter)
                r = r_l @ r_r
                h = (
                    2
                    * torch.sigmoid(torch.matmul(r, xx.unsqueeze(-1)).squeeze(-1))
                    * h
                )
            except StopIteration:
                pass
        try:
            r_l, r_r = next(r_iter)
            r = r_l @ r_r
            h = (
                2 * torch.sigmoid(torch.matmul(r, xx.unsqueeze(-1)).squeeze(-1)) * h
            )
        except StopIteration:
            pass
        finally:
            if self.deal_tuple_hidden:
                h = (h, hidden[1])
            return xx, h

    def forward(self, x, hidden=None):
        batch_size = x.size(1)
        seq_len = x.size(0)

        if hidden is None:
            if self.deal_tuple_hidden:
                hidden = (
                    torch.zeros(batch_size, self.n),
                    torch.zeros(batch_size, self.n),
                )
            else:
                hidden = torch.zeros(batch_size, self.n)
        else:
            num_dims = len(hidden.size())
            if num_dims != 2:
                assert hidden.size(0) == 1, f"Hidden state should contain 2 dimensions OR size(0)=1. Actual size is {hidden.size()!r}"
                hidden = hidden.squeeze(0)

        output = torch.empty((seq_len, batch_size, self.n))
        for seq_idx, xx in enumerate(x):

            xx, hidden = self.mogrify(xx, hidden)

            hidden = self.cell(xx, hidden)

            if self.deal_tuple_hidden:
                hidden_out = hidden[0]
            else:
                hidden_out = hidden
            output[seq_idx, ...] = hidden_out

        if self.deal_tuple_hidden:
            hidden = tuple(h.unsqueeze(0) for h in hidden)
        else:
            hidden = hidden.unsqueeze(0)
            
        return output, hidden


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):

        embedded = self.embedding(x.rename(None)).refine_names("I", "B", "H")
        output, hidden = self.rnn(embedded.rename(None))
        return output.refine_names("I", "B", "H"), hidden.refine_names(..., "B", "H")


class EncoderMogrifierRNN(EncoderRNN):
    def __init__(self, input_size, hidden_size, n_mogrifying=4):
        super(EncoderMogrifierRNN, self).__init__(input_size, hidden_size)
        self.rnn = MogrifierRNN(nn.GRUCell(hidden_size, hidden_size), n_mogrifying)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden):
        output = self.embedding(x.rename(None)).refine_names("O", "B", "H")
        output = F.relu(output)
        output, hidden = self.rnn(output.rename(None), hidden.rename(None))
        output, hidden = (
            output.refine_names("O", "B", "H"),
            hidden.refine_names(..., "B", "H"),
        )
        output = self.log_softmax(self.out(output))
        return output.refine_names(..., "V"), hidden


class DecoderMogrifierRNN(DecoderRNN):
    def __init__(self, hidden_size, output_size, n_mogrifying=4):
        super(DecoderMogrifierRNN, self).__init__(hidden_size, output_size)
        self.rnn = MogrifierRNN(nn.GRUCell(hidden_size, hidden_size), n_mogrifying)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=20):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), dim=-1)), dim=-1
        )
        attn_applied = torch.bmm(
            torch.transpose(attn_weights, 0, 1), torch.transpose(encoder_outputs, 0, 1)
        )

        output = torch.cat((embedded, attn_applied), -1)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)


def test_enc_dec():
    data.set_seed()

    batch_size = 4
    hidden_size = 100

    dataset = data.DateDataset(6)
    data_loader = data.get_date_dataloader(dataset, batch_size)

    encoder, decoder = (
        EncoderRNN(len(data.vocabulary), hidden_size),
        DecoderRNN(hidden_size, len(data.vocabulary)),
    )
    for i, o in data_loader:
        mid, hidden = encoder(i)
        print(
            decoder(
                torch.Tensor(
                    [
                        [data.vocabulary[data.__BEG__]] * i.size(1),
                        [data.vocabulary["2"]] * i.size(1),
                    ]
                ).long(),
                hidden,
            )[0].shape
        )


def test_mogrifier():
    data.set_seed()
    batch, inp, hid, seq = 7, 11, 13, 17
    n_mog = 4

    m = MogrifierRNN(nn.GRUCell(inp, hid), n_mog)

    x = torch.randn(seq, batch, inp)

    print(m(x)[0].size())


if __name__ == "__main__":
    test_mogrifier()
