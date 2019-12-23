import math
import random
import time

import torch

from torch import nn, optim

from robust_parser import data, model, config


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


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=20,
):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.device)

    loss = 0

    # for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_tensor)
    # encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[data.vocabulary[data.__BEG__]] * input_tensor.size(1)], device=config.device)

    decoder_hidden = encoder_hidden

    # use_teacher_forcing = random.random() < teacher_forcing_ratio
    use_teacher_forcing = False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        # for batch in input_tensor.size(1):
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == data.vocabulary[data.__END__]:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(
    encoder, decoder, n_iters, dataset, print_every=1000, plot_every=100, learning_rate=0.01
):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    training_pairs = [(i[0], i[2]) for i in dataset]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, iter / n_iters),
                    iter,
                    iter / n_iters * 100,
                    print_loss_avg,
                )
            )

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def main():
    data.set_seed(112)

    batch_size = 5
    hidden_size = 50
    
    dataset = data.get_date_dataloader(data.DateDataset(10**4), batch_size)
    
    encoder, decoder = (
        model.EncoderRNN(len(data.vocabulary), hidden_size),
        model.AttnDecoderRNN(hidden_size, len(data.vocabulary)),
    )
    
    train_iters(encoder, decoder, 10, dataset, 1, 1)


if __name__ == "__main__":
    main()