import datetime
import math
import random
import time

import numpy as np
import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

from robust_parser import config, data


class EncoderDecoderModel(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderDecoderModel, self).__init__()

        encoder_cls = kwargs.get("encoder_cls", EncoderRNN)
        decoder_cls = kwargs.get("decoder_cls", DecoderRNN)

        encoder_prm = kwargs.get("encoder_prm", {})
        decoder_prm = kwargs.get("decoder_prm", {})

        self.input_size = kwargs.get("input_size", len(data.vocabulary))
        self.hidden_size = kwargs.get("hidden_size", 128)
        self.output_size = kwargs.get("output_size", len(data.vocabulary))

        self.max_seq_len = kwargs.get("max_seq_len", 21)

        self.encoder = encoder_cls(self.input_size, self.hidden_size, **encoder_prm)
        self.decoder = decoder_cls(self.hidden_size, self.output_size, **decoder_prm)

        self.enc_optim = None
        self.dec_optim = None

        self.criterion = nn.NLLLoss(ignore_index=data.vocabulary[data.__PAD__])

    def _setup_clipping(self, clipping):
        """
        set up clipping
        """
        for model in [self.encoder, self.decoder]:
            for p in model.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -gradient_clip, gradient_clip)
                )

    def _l2_regularize(self, l2_penalty=10 ** -3):
        loss_reg = 0
        crit = nn.MSELoss()
        for m in [self.encoder, self.decoder]:
            for p in m.parameters():
                loss_reg += crit(p, torch.zeros_like(p))
        return l2_penalty * loss_reg

    def _teacher_forcing(self, enc_out, dec_hidden, input_mask, tgt_tensor):
        target_length = tgt_tensor.size(0)
        batch_size = tgt_tensor.size(1)
        loss = 0

        dec_input = torch.cat(
            (
                torch.full(
                    size=(1, batch_size),
                    fill_value=data.vocabulary[data.__BEG__],
                    dtype=torch.long,
                ),
                tgt_tensor[:-1, ...],
            ),
            dim=0,
        ).refine_names("O", "B")

        for seq_pos in range(target_length):
            dec_output, dec_hidden = self.decoder(
                dec_input[seq_pos, ...], dec_hidden, enc_out, input_mask
            )

            loss += self.criterion(
                dec_output.align_to("B", "V", "O").rename(None),
                tgt_tensor.align_to("B", "O").rename(None)[:, seq_pos].unsqueeze(-1),
            )
        return loss

    def _greedy_decode(
        self, enc_out, dec_hidden, input_mask, *, tgt_tensor=None, store_attn=False
    ):
        batch_size = enc_out.size(1)

        if store_attn:
            assert tgt_tensor is None
            assert hasattr(self.decoder, "attn")

        if tgt_tensor is not None:
            loss = 0
            max_seq_len = min(self.max_seq_len, tgt_tensor.size(0))
        else:
            max_seq_len = self.max_seq_len

        pred_seq = np.empty((max_seq_len, batch_size), dtype=np.int64)

        dec_input = torch.full(
            size=(1, batch_size),
            fill_value=data.vocabulary[data.__BEG__],
            dtype=torch.long,
        ).refine_names("O", "B")

        if store_attn:
            attn_l = []

        for seq_pos in range(max_seq_len):
            dec_output, dec_hidden = self.decoder(
                dec_input, dec_hidden, enc_out, input_mask
            )

            _, topi = dec_output.rename(None).topk(1, dim=-1)
            topi = topi.squeeze(-1).refine_names("O", "B")
            dec_input = topi.detach()  # detach from history as input

            if store_attn:
                attn_l.append(self.decoder.attn.alphas)

            if tgt_tensor is not None:
                loss += self.criterion(
                    dec_output.align_to("B", "V", "O").rename(None),
                    tgt_tensor.align_to("B", "O")
                    .rename(None)[:, seq_pos]
                    .unsqueeze(-1),
                )

            pred_seq[seq_pos] = dec_input.squeeze(0).numpy()

        if store_attn:
            attn_array = np.empty((len(attn_l), *attn_l[0].size()))
            for i, attn_val in enumerate(attn_l):
                attn_array[i] = attn_val.detach().numpy()

        return (
            pred_seq
            if tgt_tensor is None and not store_attn
            else (pred_seq, loss)
            if not store_attn
            else (pred_seq, attn_array)
        )

    def forward(self, x, *, mode="greedy", return_attn=False):
        """
        endcode, decode (greedy)
        """
        enc_out, enc_hidden = self.encoder(x)
        input_seq_mask = torch.ones_like(x).masked_fill(
            x == data.vocabulary[data.__PAD__], 0
        )

        if mode == "greedy":
            return self._greedy_decode(enc_out, enc_hidden, input_seq_mask, store_attn=return_attn)
        else:
            raise NotImplementedError

    def _fit_step(self, x, y, *, teacher_forcing_ratio=0.8, l2_penalty=None):
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()

        enc_output, enc_hidden = self.encoder(x)
        input_seq_mask = torch.ones_like(x).masked_fill(
            x == data.vocabulary[data.__PAD__], 0
        )

        use_teacher_forcing = random.random() < teacher_forcing_ratio

        if use_teacher_forcing:
            loss = self._teacher_forcing(
                enc_output, enc_hidden, input_seq_mask, tgt_tensor=y
            )

        else:
            _, loss = self._greedy_decode(
                enc_output, enc_hidden, input_seq_mask, tgt_tensor=y
            )

        if l2_penalty is not None:
            loss += self._l2_regularize(l2_penalty)

        loss.backward()

        self.enc_optim.step()
        self.dec_optim.step()

        return loss.item()

    def fit(self, dataset_train, dataset_test, **kwargs):
        """
        trains over the dataset_train with some parameters
        """
        n_epochs = kwargs.get("n_epochs", 10)
        learning_rate = kwargs.get("learning_rate", 10 ** -3)
        teacher_forcing_ratio = kwargs.get("teacher_forcing_ratio", 0.5)
        l2_penalty = kwargs.get("l2_penalty", None)

        eval_every = kwargs.get("eval_every", 5)
        verbose = kwargs.get("verbose", False)

        optimizer = kwargs.get("optimizer", "Adam")
        optimizer = getattr(optim, optimizer, "Adam")

        self.enc_optim = optimizer(self.encoder.parameters(), lr=learning_rate)
        self.dec_optim = optimizer(self.decoder.parameters(), lr=learning_rate)

        print_loss_total = 0.0
        start = time.time()

        for epoch in range(n_epochs):
            dataset_train.redraw_dataset()
            for x, y in dataset_train:

                decode_loss = self._fit_step(
                    x,
                    y,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    l2_penalty=l2_penalty,
                )
                print_loss_total += decode_loss

            print_loss_avg = print_loss_total / len(dataset_train)
            print_loss_total = 0
            print(
                f"[{epoch + 1:3}] {(epoch+1)/n_epochs*100:5.2f}% ({datetime.timedelta(seconds=int(time.time()-start))!s}) | Loss={print_loss_avg:.5f}"
            )

            if (epoch + 1) % eval_every == 0:
                print(self.evaluate(dataset_test, verbose=verbose))

    def evaluate(self, dataset, **kwargs):
        """
        eval metrics over dataset
        """
        verbose = kwargs.get("verbose", False)

        self.eval()

        with torch.no_grad():
            errors = []
            tp_count = 0
            loss = 0
            for x, y in dataset:
                batch_size = x.size(1)
                encoder_output, encoder_hidden = self.encoder(x)
                input_mask = torch.ones_like(x).masked_fill(
                    x == data.vocabulary[data.__PAD__], 0
                )
                pred_seq, decode_loss = self._greedy_decode(
                    encoder_output, encoder_hidden, input_mask=input_mask, tgt_tensor=y
                )

                loss += decode_loss

                for batch_idx in range(batch_size):

                    if any(
                        p != yy.item()
                        for p, yy in zip(pred_seq[:, batch_idx], y[:, batch_idx])
                    ):
                        if verbose:
                            errors.append(
                                (
                                    f"PRD : {''.join(data.rev_vocabulary[i] for i in pred_seq[:, batch_idx])}",
                                    f"TGT : {''.join(data.rev_vocabulary[i.item()] for i in y[:, batch_idx])}",
                                    f"RAW : {''.join(data.rev_vocabulary[i.item()] for i in x[:, batch_idx])}",
                                )
                            )
                    else:
                        tp_count += 1

            accuracy = tp_count / dataset.n_elements
            loss = loss.item() / dataset.n_elements
            if verbose:
                print(
                    *(
                        "\n".join(i)
                        for i in random.choices(errors, k=min(10, len(errors)))
                    ),
                    sep="\n\n",
                )
                print(f"[ACCURACY]: {accuracy}")
                print(f"[LOSS]: {loss}")

            return {"loss": loss, "perplexity": math.exp(loss), "accuracy": accuracy}


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

    def forward(self, x, hidden, encoder_outputs=None, input_mask=None):
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
