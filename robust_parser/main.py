import datetime
import math
import random
import time

import torch

from torch import nn, optim

from robust_parser import data, model, config
from robust_parser.model_lab import mogrifier

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


teacher_forcing_ratio = 0.8


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
            target_tensor[:-1, ...],
        ),
        dim=0,
    ).refine_names("O", "B")
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

    loss += criterion(
        decoder_output.align_to("B", "V", "O").rename(None),
        target_tensor.align_to("B", "O").rename(None),
    )

    return loss


def greedy_decode(decoder, decoder_input, decoder_hidden, criterion, target_tensor):
    target_length = target_tensor.size(0)
    pred_seq = []
    loss = 0
    for seq_pos in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        topv, topi = decoder_output.rename(None).topk(1, dim=-1)
        topi = topi.squeeze(-1).refine_names("O", "B")
        decoder_input = topi.detach()  # detach from history as input

        loss += criterion(
            decoder_output.align_to("B", "V", "O").rename(None),
            target_tensor.align_to("B", "O").rename(None)[:, seq_pos].unsqueeze(-1),
        )
        pred_seq.append(decoder_input.squeeze().numpy())

    return loss, pred_seq


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

    loss, _ = greedy_decode(decoder, decoder_input_b, decoder_hidden_b, criterion, target_tensor)

    return loss


def L2_regularize(model_l, l2_penalty=10**-4):
    loss_reg = 0
    crit = nn.MSELoss()
    for m in model_l:
        for p in m.parameters():
            loss_reg += crit(p, torch.zeros_like(p))
    return l2_penalty * loss_reg


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

    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        loss = teacher_forcing(
            decoder_hidden, encoder_output, target_tensor, decoder, criterion
        )

    else:
        loss = greedy_decode_training(
            decoder_hidden, encoder_output, target_tensor, decoder, criterion
        )

    loss += L2_regularize([encoder, decoder])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def train_iters(
    encoder,
    decoder,
    n_epochs,
    dataset: data.DateLoader,
    dataset_test: data.DateLoader,
    *,
    eval_every=2,
    learning_rate=0.003
):
    start = time.time()

    encoder.train()
    decoder.train()

    gradient_clip = 10
    if gradient_clip is not None:
        for model in [encoder, decoder]:
            for p in model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -gradient_clip, gradient_clip))

    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(ignore_index=data.vocabulary[data.__PAD__])

    for epoch in range(n_epochs):
        dataset.redraw_dataset()
        for x, y in tqdm.tqdm(dataset):

            loss = train(
                x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
            )
            print_loss_total += loss
            plot_loss_total += loss

        print_loss_avg = print_loss_total / len(dataset)
        print_loss_total = 0
        print(
            f"[{epoch + 1}] {(epoch+1)/n_epochs*100:4.2f}% ({datetime.timedelta(seconds=int(time.time()-start))!s}) | Loss={print_loss_avg:.5f}"
        )

        if (epoch + 1) % eval_every == 0:
            evaluate(encoder, decoder, dataset_test, criterion, verbose=True)

    return evaluate(encoder, decoder, dataset_test, criterion)
    #     if iter % plot_every == 0:
    #         plot_loss_avg = plot_loss_total / plot_every
    #         plot_losses.append(plot_loss_avg)
    #         plot_loss_total = 0

    # showPlot(plot_losses)


def evaluate(encoder, decoder, dataset, criterion, *, verbose=False):
    encoder.eval()
    decoder.eval()
    
    rev_vocabulary = list(data.vocabulary.keys())

    with torch.no_grad():
        errors = []
        count_ok = 0
        loss = 0
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

            local_loss, pred_seq = greedy_decode(decoder, decoder_input_b, decoder_hidden_b, criterion, y)

            loss += local_loss
            
            if any(p != yy.item() for p, yy in zip(pred_seq, y)):
                if verbose:
                    errors.append((f"PRD : {''.join(rev_vocabulary[i] for i in pred_seq)}",
                                   f"TGT : {''.join(rev_vocabulary[i.item()] for i in y)}",
                                   f"RAW : {''.join(rev_vocabulary[i.item()] for i in x)}",
                                   "\n"))
            else:
                count_ok += 1
        print(*("\n".join(i) for i in random.choices(errors, k=min(10, len(errors)))))
        print(f"[ACCURACY]: {count_ok / len(dataset)}")
        print(f"[LOSS]: {loss / len(dataset)}")
        
        return loss.item()

     
def train_eval(parameters):
    learning_rate = parameters["learning_rate"]
    hidden_size = parameters["hidden_size"]
    n_mogrify = parameters["n_mogrify"]
    
    data.set_seed()

    train_data_size = 5 * 10 ** 3
    test_data_size = 10 ** 3

    batch_size = 64

    dataset_train = data.get_date_dataloader(
        data.DateDataset(train_data_size), batch_size
    )
    dataset_test = data.get_date_dataloader(data.DateDataset(test_data_size), 1)

    encoder, decoder = (à
        mogrifier.EncoderMogrifierRNN(len(data.vocabulary), hidden_size, n_mogrify),
        # model.EncoderRNN(len(data.vocabulary), hidden_size),
        mogrifier.DecoderMogrifierRNN(hidden_size, len(data.vocabulary), n_mogrify),
        # model.DecoderRNN(hidden_size, len(data.vocabulary)),
    )

    return train_iters(encoder, decoder, 10, dataset_train, dataset_test, eval_every=20, learning_rate=learning_rate)
    


def main():
    data.set_seed()

    train_data_size = 5 * 10 ** 3
    test_data_size = 10 ** 3

    batch_size = 64
    hidden_size = 256

    

    dataset_train = data.get_date_dataloader(
        data.DateDataset(train_data_size), batch_size
    )
    dataset_test = data.get_date_dataloader(data.DateDataset(test_data_size), 1)

    encoder, decoder = (
        mogrifier.EncoderMogrifierRNN(len(data.vocabulary), hidden_size, 6),
        # model.EncoderRNN(len(data.vocabulary), hidden_size),
        mogrifier.DecoderMogrifierRNN(hidden_size, len(data.vocabulary), 6),
        # model.DecoderRNN(hidden_size, len(data.vocabulary)),
    )

    train_iters(encoder, decoder, 5, dataset_train, dataset_test)

    return encoder, decoder


if __name__ == "__main__":
    # main()
    # train_eval({"learning_rate": 0.001, "n_mogrify": 3, "hidden_size": 32})
    
    from ax.plot.contour import plot_contour
    from ax.plot.trace import optimization_trace_single_method
    from ax.service.managed_loop import optimize
    from ax.utils.notebook.plotting import render, init_notebook_plotting
    
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "learning_rate", "value_type": "float", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "n_mogrify", "value_type": "int", "type": "range", "bounds": [0, 10]},
            {"name": "hidden_size", "value_type": "int", "type": "range", "bounds": [10, 256]},
        ],
        evaluation_function=train_eval,
        objective_name='loss',
        minimize=True,
        total_trials=10
    )
    
    print(best_parameters, values, experiment, model)
