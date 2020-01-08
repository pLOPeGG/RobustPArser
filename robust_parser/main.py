import datetime
import math
import random
import time

import torch

from torch import nn, optim

from robust_parser import data, model, config

import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.switch_backend("agg")


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


teacher_forcing_ratio = 0.5


def teacher_forcing(decoder_hidden, encoder_output, target_tensor, decoder, criterion):
    loss = 0
    target_length = target_tensor.size(0)
    batch_size = target_tensor.size(1)

    decoder_input = torch.cat(
        (
            torch.full(
                size=(1, batch_size),
                fill_value=data.vocabulary[data.__BEG__],
                dtype=torch.long,
            ),
            target_tensor,
        ),
        dim=0,
    ).refine_names("O", "B")
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

    loss += criterion(
        decoder_output[:-1, ...].align_to("B", "V", "O").rename(None),
        target_tensor.align_to("B", "O").rename(None),
    )

    return loss


def greedy_decode_training(
    decoder_hidden, encoder_output, target_tensor, decoder, criterion
):
    loss = 0
    target_length = target_tensor.size(0)
    batch_size = target_tensor.size(1)

    decoder_input_b = torch.full(
        size=(1, batch_size), fill_value=data.vocabulary[data.__BEG__], dtype=torch.long
    )
    decoder_hidden_b = decoder_hidden

    for seq_pos in range(target_length):
        decoder_output, decoder_hidden_b = decoder(decoder_input_b, decoder_hidden_b)

        topv, topi = decoder_output.rename(None).topk(1, dim=-1)
        topi = topi.squeeze(-1).refine_names("O", "B")
        decoder_input_b = topi.detach()  # detach from history as input

        loss += criterion(
            decoder_output.align_to("B", "V", "O").rename(None),
            target_tensor.align_to("B", "O").rename(None)[:, seq_pos].unsqueeze(-1),
        )

    return loss


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=20
):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_output, encoder_hidden = encoder(input_tensor)

    # encoder_outputs = torch.cat(
    #     (
    #         encoder_outputs,
    #         torch.zeros(
    #             (max_length - input_length, batch_size, encoder_outputs.size(2))
    #         ),
    #     ),
    #     dim=0,
    # )

    decoder_hidden = encoder_hidden

    # use_teacher_forcing = random.random() < teacher_forcing_ratio
    use_teacher_forcing = True

    if use_teacher_forcing:
        loss = teacher_forcing(
            decoder_hidden, encoder_output, target_tensor, decoder, criterion
        )

    else:
        loss = greedy_decode_training(
            decoder_hidden, encoder_output, target_tensor, decoder, criterion
        )

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def train_iters(
    encoder,
    decoder,
    n_epochs,
    dataset: data.DateLoader,
    plot_every=1000,
    learning_rate=0.003,
    gradient_clip=10
):
    start = time.time()

    encoder.train()
    decoder.train()

    for model in [encoder, decoder]:
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -gradient_clip, gradient_clip))

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(ignore_index=data.vocabulary[data.__PAD__])

    for epoch in range(n_epochs):
        for x, y in tqdm.tqdm(dataset):

            loss = train(
                x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
            )
            print_loss_total += loss
            plot_loss_total += loss

        print_loss_avg = print_loss_total / len(dataset)
        print_loss_total = 0
        print(
            f"[{epoch + 1}] {(epoch+1)/n_epochs*100:.2f}% ({datetime.timedelta(seconds=int(time.time()-start))!s}) | Loss={print_loss_avg:.5f}"
        )

    #     if iter % plot_every == 0:
    #         plot_loss_avg = plot_loss_total / plot_every
    #         plot_losses.append(plot_loss_avg)
    #         plot_loss_total = 0

    # showPlot(plot_losses)


def evaluate(encoder, decoder, dataset):
    encoder.eval()
    decoder.eval()
    
    rev_vocabulary = list(data.vocabulary.keys())

    with torch.no_grad():
        count_ok = 0
        for x, y in dataset:
            target_length = y.size(0)
            batch_size = y.size(1)

            encoder_output, encoder_hidden = encoder(x)
            decoder_hidden = encoder_hidden
            decoder_input_b = torch.full(
                size=(1, batch_size),
                fill_value=data.vocabulary[data.__BEG__],
                dtype=torch.long,
            )
            decoder_hidden_b = decoder_hidden

            pred_seq = []
            for seq_pos in range(target_length):
                decoder_output, decoder_hidden_b = decoder(
                    decoder_input_b, decoder_hidden_b
                )

                topv, topi = decoder_output.rename(None).topk(1, dim=-1)
                topi = topi.squeeze(-1).refine_names("O", "B")
                decoder_input_b = topi.detach()  # detach from history as input

                pred_seq.append(decoder_input_b.item())

            if any(p != yy.item() for p, yy in zip(pred_seq, y)):
                print(f"PRD : {''.join(rev_vocabulary[i] for i in pred_seq)}")
                print(f"TGT : {''.join(rev_vocabulary[i.item()] for i in y)}")
                print(f"RAW : {''.join(rev_vocabulary[i.item()] for i in x)}")
                print()
            else:
                count_ok += 1
                
        print(f"[ACCURACY]: {count_ok / len(dataset)}")


def main():
    data.set_seed()

    train_data_size = 10 ** 4
    test_data_size = 10 ** 3

    batch_size = 64
    hidden_size = 128
    
    gradient_clip = 10

    dataset_train = data.get_date_dataloader(
        data.DateDataset(train_data_size), batch_size
    )
    dataset_test = data.get_date_dataloader(data.DateDataset(test_data_size), 1)

    encoder, decoder = (
        model.EncoderRNN(len(data.vocabulary), hidden_size),
        model.DecoderMogrifierRNN(hidden_size, len(data.vocabulary)),
    )

    train_iters(encoder, decoder, 10, dataset_train)

    evaluate(encoder, decoder, dataset_test)


if __name__ == "__main__":
    main()
