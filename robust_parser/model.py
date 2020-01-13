import torch

from torch import nn
from torch.nn import functional as F

from robust_parser import config, data


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)

    def forward(self, x):

        embedded = self.embedding(x.rename(None)).refine_names("I", "B", "H")
        output, hidden = self.rnn(embedded.rename(None))
        if isinstance(hidden, tuple):  # LSTM like rnn
            hidden = (
                hidden[0].refine_names(..., "B", "H"),
                hidden[1].refine_names(..., "B", "H"),
            )
        else:
            hidden = hidden.refine_names(..., "B", "H")
        return output.refine_names("I", "B", "H"), hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, share_weights=True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.share_weights = share_weights

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        if self.share_weights:
            self.out.weight = nn.Parameter(self.embedding.weight)

    def forward(self, x, hidden):
        output = self.embedding(x.rename(None)).refine_names("O", "B", "H")
        output = F.relu(output)
        if isinstance(hidden, tuple):  # LSTM like rnn
            hidden = (hidden[0].rename(None), hidden[1].rename(None))
        else:
            hidden = hidden.rename(None)
        output, hidden = self.rnn(output.rename(None), hidden)

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


if __name__ == "__main__":
    test_enc_dec()
