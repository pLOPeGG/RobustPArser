import torch
import torch.nn.functional as F

from torch import nn

from robust_parser import model, data, config


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

        self.k = max(20, min(self.m, self.n) // 16)

        self.init_weights()

    def init_weights(self):
        # fmt: off
        self.Q_left  = nn.ParameterList([nn.Parameter(torch.empty((self.m, self.k), dtype=torch.float, requires_grad=True))
                                         for _ in range(1, self.n_mogrifying, 2)])
        self.Q_right = nn.ParameterList([nn.Parameter(torch.empty((self.k, self.n), dtype=torch.float, requires_grad=True))
                                         for _ in range(1, self.n_mogrifying, 2)])
        self.R_left  = nn.ParameterList([nn.Parameter(torch.empty((self.n, self.k), dtype=torch.float, requires_grad=True))
                                         for _ in range(2, self.n_mogrifying, 2)])
        self.R_right = nn.ParameterList([nn.Parameter(torch.empty((self.k, self.m), dtype=torch.float, requires_grad=True))
                                         for _ in range(2, self.n_mogrifying, 2)])
        # fmt: on

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
                h = 2 * torch.sigmoid(torch.matmul(r, xx.unsqueeze(-1)).squeeze(-1)) * h
            except StopIteration:
                pass
        try:
            r_l, r_r = next(r_iter)
            r = r_l @ r_r
            h = 2 * torch.sigmoid(torch.matmul(r, xx.unsqueeze(-1)).squeeze(-1)) * h
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
                assert (
                    hidden.size(0) == 1
                ), f"Hidden state should contain 2 dimensions OR size(0)=1. Actual size is {hidden.size()!r}"
                if self.deal_tuple_hidden:
                    hidden = (hidden.squeeze(0), torch.zeros_like(hidden.squeeze(0)))
                else:
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


class MogrifierRNNShort(MogrifierRNN):
    def __init__(self, rnn_cell, n_mogrifying, reduce_params=True):
        self.reduce_params = reduce_params
        super(MogrifierRNNShort, self).__init__(rnn_cell, n_mogrifying)

    def init_weights(self):
        if self.reduce_params:
            # fmt: off
            self.Q_left  = nn.Parameter(torch.empty((self.m, self.k), dtype=torch.float, requires_grad=True))
            self.Q_right = nn.Parameter(torch.empty((self.k, self.n), dtype=torch.float, requires_grad=True))
            self.R_left  = nn.Parameter(torch.empty((self.n, self.k), dtype=torch.float, requires_grad=True))
            self.R_right = nn.Parameter(torch.empty((self.k, self.m), dtype=torch.float, requires_grad=True))
            # fmt: on

            for w in [self.Q_left, self.Q_right, self.R_left, self.R_right]:
                nn.init.xavier_uniform_(w)

        else:
            # fmt: off
            self.Q = nn.Parameter(torch.empty((self.m, self.n), dtype=torch.float, requires_grad=True))
            self.R = nn.Parameter(torch.empty((self.n, self.m), dtype=torch.float, requires_grad=True))
            # fmt: on

            for w in [self.Q, self.R]:
                nn.init.xavier_uniform_(w)

    def mogrify(self, xx, hidden):
        if self.deal_tuple_hidden:
            h = hidden[0]
        else:
            h = hidden

        if self.reduce_params:
            Q, R = self.Q_left @ self.Q_right, self.R_left @ self.R_right

        else:
            Q, R = self.Q, self.R

        for i in range(1, self.n_mogrifying + 1):
            if i % 2 == 0:
                h = (
                    2 * torch.sigmoid(torch.matmul(R, xx.unsqueeze(-1)).squeeze(-1))
                ) * h
            else:
                xx = (
                    2 * torch.sigmoid(torch.matmul(Q, h.unsqueeze(-1)).squeeze(-1))
                ) * xx

        if self.deal_tuple_hidden:
            h = (h, hidden[1])
        return xx, h


class EncoderMogrifierRNN(model.EncoderRNN):
    def __init__(self, input_size, hidden_size, n_mogrifying=5):
        super(EncoderMogrifierRNN, self).__init__(input_size, hidden_size)
        # self.rnn = MogrifierRNN(nn.LSTMCell(hidden_size, hidden_size), n_mogrifying)
        self.rnn = MogrifierRNNShort(
            nn.LSTMCell(hidden_size, hidden_size), n_mogrifying
        )
        # self.rnn = MogrifierRNNGit(hidden_size, hidden_size, n_mogrifying)


class DecoderMogrifierRNN(model.DecoderRNN):
    def __init__(self, hidden_size, output_size, n_mogrifying=5):
        super(DecoderMogrifierRNN, self).__init__(hidden_size, output_size)
        # self.rnn = MogrifierRNN(nn.LSTMCell(hidden_size, hidden_size), n_mogrifying)
        self.rnn = MogrifierRNNShort(
            nn.LSTMCell(hidden_size, hidden_size), n_mogrifying
        )
        # self.rnn = MogrifierRNNGit(hidden_size, hidden_size, n_mogrifying)


class MogrifierLSTMCellGit(nn.Module):
    # Taken from https://github.com/fawazsammani/mogrifier-lstm-pytorch as is
    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCellGit, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.q = nn.Linear(hidden_size, input_size)
        self.r = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.init_parameters()
        self.mogrify_steps = mogrify_steps

    def init_parameters(self):
        std = 1.0 / self.hidden_size ** 0.5
        for p in self.parameters():
            p.data.uniform_(-std, std)

    def mogrify(self, x, h):
        for i in range(1, self.mogrify_steps + 1):
            if i % 2 == 0:
                h = (2 * torch.sigmoid(self.r(x))) * h
            else:
                x = (2 * torch.sigmoid(self.q(h))) * x
        return x, h

    def forward(self, x, states):
        """
        inp shape: (batch_size, input_size)
        each of states shape: (batch_size, hidden_size)
        """
        ht, ct = states
        x, ht = self.mogrify(x, ht)  # Note: This should be called every timestep
        gates = self.x2h(x) + self.h2h(ht)  # (batch_size, 4 * hidden_size)
        in_gate, forget_gate, new_memory, out_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        new_memory = self.tanh(new_memory)
        c_new = (forget_gate * ct) + (in_gate * new_memory)
        h_new = out_gate * self.tanh(c_new)

        return h_new, c_new


class MogrifierRNNGit(nn.Module):
    # Taken from https://github.com/fawazsammani/mogrifier-lstm-pytorch and adapted to our situation.
    def __init__(self, input_size, hidden_size, n_mogrifying):
        super(MogrifierRNNGit, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_mogrifying = n_mogrifying

        self.rnn_cell = MogrifierLSTMCellGit(input_size, hidden_size, n_mogrifying)

    def forward(self, x, hidden=None):
        batch_size = x.size(1)
        seq_len = x.size(0)

        if hidden is None:

            hidden = (
                torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size),
            )

        else:
            num_dims = len(hidden.size())
            if num_dims != 2:
                assert (
                    hidden.size(0) == 1
                ), f"Hidden state should contain 2 dimensions OR size(0)=1. Actual size is {hidden.size()!r}"
                hidden = (hidden.squeeze(0), torch.zeros_like(hidden.squeeze(0)))

        output = torch.empty((seq_len, batch_size, self.hidden_size))
        for seq_idx, xx in enumerate(x):

            hidden = self.rnn_cell(xx, hidden)

            hidden_out = hidden[0]

            output[seq_idx, ...] = hidden_out

        hidden = tuple(h.unsqueeze(0) for h in hidden)
        return output, hidden


def test_mogrifier():
    data.set_seed()
    batch, inp, hid, seq = 7, 11, 13, 17
    n_mog = 4

    m = MogrifierRNNGit(inp, hid, n_mog)

    x = torch.randn(seq, batch, inp)

    print(m(x)[0].size())
    print(m)


if __name__ == "__main__":
    test_mogrifier()
