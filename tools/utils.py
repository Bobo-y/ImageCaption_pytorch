import torch
import cv2
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch, paded_token=6688):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    descriptions = list(items[1])
    max_len = 0
    for description in descriptions:
        max_len = max(max_len, len(description))

    labels = torch.zeros(max_len, len(batch), dtype=torch.long).fill_(paded_token)
    for idx, description in enumerate(descriptions):
        cur_len = len(description)
        labels[:cur_len, idx] = torch.tensor(description)
    items[1] = labels
    return items


def process_img(img, height, width, transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    out_img = transform(img)
    return torch.unsqueeze(out_img, 0)


def decode_str(idx, vocab):
    str_re = ""
    for i in range(1, len(idx) - 1):
        str_re += " " + vocab[idx[i]]
    return str_re
