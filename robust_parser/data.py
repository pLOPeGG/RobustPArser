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

ISODate = NewType("ISODate", str)
Date = NewType("Date", str)

charset = "1234567890-/. "

vocabulary = {
    __BEG__: 0,
    __END__: 1,
    __PAD__: 2,
    **{c: i for i, c in enumerate(charset, 3)},
}


class DateDataset(torch.utils.data.Dataset):
    """Some Information about DateDataset"""

    def __init__(self, n=100, seed=None):
        super(DateDataset, self).__init__()
        if seed is not None:
            random.seed(seed)

        self._raw_dataset = [generate_date_pair() for _ in range(n)]
        self._dataset = [
            (
                [vocabulary[__BEG__]] + [vocabulary[c] for c in x],
                [vocabulary[c] for c in y] + [vocabulary[__END__]],
            )
            for x, y in self._raw_dataset
        ]

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


class DateLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, drop_last=False):
        self._dataset = dataset
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

        for next_batch in self._sampler:
            max_in_len, max_out_len = (
                max(len(self._dataset[i][j]) for j in next_batch) for i in range(2)
            )
            
            """next_input = torch.ones(max_seq_len,
                                    self.batch_size,
                                    dtype=torch.int64).fill_(pad_item)

            next_target = torch.ones(max_seq_len + 1,
                                     self.batch_size,
                                     dtype=torch.int64).fill_(pad_item)
            next_target[0, :] = beg_item

            next_output = torch.ones(max_seq_len + 1,
                                     self.batch_size,
                                     dtype=torch.int64).fill_(pad_item)

            # ? If too much memory is used, merge output and target and move
            # ? further processing to the training loop
            for i, tensor_idx in enumerate(next_batch):
                tensor = torch.Tensor(self._dataset[tensor_idx])
                next_input[:len(tensor), i] = tensor

                next_target[1:len(tensor) + 1, i] = tensor

                next_output[:len(tensor), i] = tensor
                next_output[len(tensor), i] = end_item
                if self.mask:
                    rnd_mask_idx = self.mask_index_sample(len(tensor))
                    next_input[rnd_mask_idx, i] = msk_item

            yield (next_input.to(self.device), next_target.to(self.device),
                   next_output.to(self.device))
            """

    def __len__(self):
        if self.drop_last:
            return len(self._sampler) // self.batch_size
        else:
            return (len(self._sampler) + self.batch_size - 1) // self.batch_size


def get_date_dataloader(
    dataset: DateDataset, batch_size: int
) -> torch.utils.data.DataLoader:
    sampler = torch.utils.data.RandomSampler(dataset)
    return PaddedBatchSampler(dataset, batch_size=batch_size, sampler=sampler)


def generate_date() -> datetime.date:
    # up to 20 years
    days_diff = datetime.timedelta(days=random.randint(0, 365 * 20 + 5))
    beg_date = datetime.date(2000, 1, 1)

    return beg_date + days_diff


def random_stringify_date(date: datetime.date) -> Date:
    sep_rnd = random.choice("-/. ")

    pad_rnd = True if random.random() < 0.5 else False

    if pad_rnd:
        return f"{date.year}{sep_rnd}{date.month:02}{sep_rnd}{date.day:02}"
    else:
        return f"{date.year}{sep_rnd}{date.month}{sep_rnd}{date.day}"


def generate_date_pair() -> Tuple[Date, ISODate]:
    iso_date = generate_date()

    str_date = random_stringify_date(iso_date)

    return str_date, str(iso_date)


def set_seed(seed=10):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)


if __name__ == "__main__":
    set_seed()
    dataset = DateDataset(4)
    data_loader = get_date_dataloader(dataset, 3)
    for i in data_loader:
        print(i)
