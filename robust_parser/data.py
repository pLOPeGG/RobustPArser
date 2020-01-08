import datetime
import random
import re
import time

from typing import Tuple, NewType

import torch

from robust_parser import config


__BEG__ = "__BEG__"
__END__ = "__END__"
__PAD__ = "__PAD__"
__MSK__ = "__MSK__"

ISODate = NewType("ISODate", str)
Date = NewType("Date", str)

charset = "1234567890-/. "

vocabulary = {
    __BEG__: 0,
    __END__: 1,
    __PAD__: 2,
    __MSK__: 3,
    **{c: i for i, c in enumerate(charset, 4)},
}


class DateDataset(torch.utils.data.Dataset):
    """Some Information about DateDataset"""

    def __init__(self, n=100, seed=None):
        super(DateDataset, self).__init__()
        if seed is not None:
            random.seed(seed)

        self._raw_dataset = [generate_date_pair() for _ in range(n)]
        self._dataset = [
            ([vocabulary[c] for c in x], [vocabulary[c] for c in y],)
            for x, y in self._raw_dataset
        ]

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


class DateLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, drop_last=False):
        self._dataset = dataset

        self.device = config.device
        self.batch_size = batch_size
        self.drop_last = drop_last

        self._sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(self._dataset, replacement=False),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )

    def __iter__(self):
        beg_item = vocabulary[__BEG__]
        end_item = vocabulary[__END__]
        pad_item = vocabulary[__PAD__]
        msk_item = vocabulary[__MSK__]

        for next_batch in self._sampler:
            max_in_len, max_out_len = (
                max(len(self._dataset[j][i]) for j in next_batch) for i in range(2)
            )

            next_input = torch.ones(
                max_in_len, len(next_batch), dtype=torch.int64, names=("I", "B")
            ).fill_(pad_item)

            # next_target = torch.ones(max_out_len + 1,
            #                          len(next_batch),
            #                          dtype=torch.int64).fill_(pad_item)
            # next_target[0, :] = beg_item

            next_output = torch.ones(
                max_out_len + 1, len(next_batch), names=("O", "B"), dtype=torch.int64
            ).fill_(pad_item)

            # ? If too much memory is used, merge output and target and move
            # ? further processing to the training loop
            for i, tensor_idx in enumerate(next_batch):
                tensor_in, tensor_out = (
                    torch.Tensor(i) for i in self._dataset[tensor_idx]
                )
                next_input[: len(tensor_in), i] = tensor_in

                # next_target[1:len(tensor_out) + 1, i] = tensor_out

                next_output[: len(tensor_out), i] = tensor_out
                next_output[len(tensor_out), i] = end_item

            yield (next_input.to(self.device), next_output.to(self.device))

    def __len__(self):
        return len(self._sampler)


def get_date_dataloader(
    dataset: DateDataset, batch_size: int
) -> torch.utils.data.DataLoader:
    return DateLoader(dataset, batch_size=batch_size, drop_last=True)


def generate_date() -> datetime.date:
    # up to 20 years
    days_diff = datetime.timedelta(days=random.randint(0, 365 * 20 + 5))
    beg_date = datetime.date(2000, 1, 1)

    return beg_date + days_diff


def random_stringify_date(date: datetime.date) -> Date:
    sep_rnd = random.choice("-/. ")

    pad_rnd = "" if random.random() < 0.1 else "02"
    fmt_rnd = random.choice([
        "{date.year}{sep_rnd}{date.month:{pad_rnd}}{sep_rnd}{date.day:{pad_rnd}}",
        "{date.day:{pad_rnd}}{sep_rnd}{date.month:{pad_rnd}}{sep_rnd}{date.year}"
    ])

    return fmt_rnd.format(date=date, pad_rnd=pad_rnd, sep_rnd=sep_rnd)


def generate_date_pair() -> Tuple[Date, ISODate]:
    iso_date = generate_date()

    str_date = random_stringify_date(iso_date)

    return str_date, str(iso_date)


def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        seed = int(time.time_ns())
        print(seed)
        random.seed(seed)
        torch.manual_seed(seed)


if __name__ == "__main__":
    set_seed()
    dataset = DateDataset(6)
    data_loader = get_date_dataloader(dataset, 5)
    print(dataset._raw_dataset)
    for i in data_loader:
        print(i)
        print("".join(list(vocabulary.keys())[i[0][j, 0]] for j in range(len(i[0]))))
