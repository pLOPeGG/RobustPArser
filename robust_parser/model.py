import torch

from torch import nn
from torch.nn import functional as F

from robust_parser import config, data


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):

        embedded = self.embedding(x.rename(None)).refine_names("I", "B", "H")
        output, hidden = self.rnn(embedded.rename(None))
        if isinstance(hidden, tuple):  # LSTM like rnn
            hidden, carry = hidden
        return output.refine_names("I", "B", "H"), hidden.refine_names(..., "B", "H")


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
        
        if isinstance(hidden, tuple):  # LSTM like rnn
            hidden, carry = hidden
        
        output, hidden = (
            output.refine_names("O", "B", "H"),
            hidden.refine_names(..., "B", "H"),
        )
        output = self.log_softmax(self.out(output))
        return output.refine_names(..., "V"), hidden


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



if __name__ == "__main__":
    test_enc_dec()
