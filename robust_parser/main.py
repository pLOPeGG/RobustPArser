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


def train_eval(parameters):
    data.set_seed()
    encoder_cls = parameters.get("encoder_cls", model.EncoderRNN)
    decoder_cls = parameters.get("decoder_cls", model.DecoderRNN)

    encoder_prm = parameters.get("encoder_prm", {})
    decoder_prm = parameters.get("decoder_prm", {})

    hidden_size = parameters.get("hidden_size", 128)

    m = model.EncoderDecoderModel(
        encoder_cls=encoder_cls,
        decoder_cls=decoder_cls,
        encoder_prm=encoder_prm,
        decoder_prm=decoder_prm,
        hidden_size=hidden_size,
    )

    batch_size = parameters.get("batch_size", 32)

    dataset_train = data.get_date_dataloader(data.DateDataset(10 ** 3), batch_size)
    dataset_test = data.get_date_dataloader(data.DateDataset(10 ** 4), batch_size)

    learning_rate = parameters.get("learning_rate", 10 ** -3)
    n_epochs = parameters.get("n_epochs", 30)
    teacher_forcing_ratio = parameters.get("teacher_forcing_ratio", 0.5)
    l2_penalty = parameters.get("l2_penalty", None)

    optimizer = parameters.get("optimizer", "Adam")

    m.fit(
        dataset_train,
        dataset_test,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        teacher_forcing_ratio=teacher_forcing_ratio,
        l2_penalty=l2_penalty,
        optimizer=optimizer,
    )

    evaluation = m.evaluate(dataset_test)
    print(parameters, evaluation, sep='\n')
    return evaluation["loss"]


def hyper_opt():
    from ax.service.managed_loop import optimize
    # from ax import optimize
    
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "learning_rate",
                "value_type": "float",
                "type": "range",
                "bounds": [1e-6, 0.1],
                "log_scale": True,
            },
            {
                "name": "teacher_forcing_ratio",
                "value_type": "float",
                "type": "range",
                "bounds": [0.0, 1.0]
            },
            {
                "name": "l2_penalty",
                "value_type": "float",
                "type": "range",
                "bounds": [1e-6, 1.0],
                "log_scale": True
            },
            {
                "name": "n_mogrify",
                "value_type": "int",
                "type": "range",
                "bounds": [0, 10],
            },
            {
                "name": "hidden_size",
                "value_type": "int",
                "type": "range",
                "bounds": [10, 512],
            },
        ],
        evaluation_function=train_eval,
        objective_name="loss",
        minimize=True,
        total_trials=20,
    )
    return best_parameters, values, experiment, model


def main():
    hyper_opt()


if __name__ == "__main__":
    main()
