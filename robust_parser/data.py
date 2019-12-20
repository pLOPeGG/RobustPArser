import datetime
import random
import re
import time

from typing import Tuple, NewType

import torch

from robust_parser import config

seed = 10
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)

__BEG__ = "__BEG__"
__END__ = "__END__"

ISODate = NewType("ISODate", str)
Date = NewType("Date", str)

charset = "1234567890-/. "

vocabulary = {__BEG__: 0, __END__: 1, **{c: i for i, c in enumerate(charset, 2)}}


class DateDataset(torch.utils.data.Dataset):
    """Some Information about DateDataset"""

    def __init__(self, n=100, seed=None):
        super(DateDataset, self).__init__()
        if seed is not None:
            random.seed(seed)

        self._dataset = [generate_date_pair() for _ in range(n)]

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def get_date_dataloader(
    dataset: DateDataset, batch_size: int
) -> torch.utils.data.DataLoader:
    sampler = torch.utils.data.RandomSampler(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)


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


if __name__ == "__main__":
    dataset = DateDataset(10)
    data_loader = get_date_dataloader(dataset, 3)
    for i in data_loader:
        print(i)
