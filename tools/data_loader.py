import torch
import cv2
import os
import string
import random
import numpy as np
from torch.utils.data import Dataset


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
"""
Flickr8k image caption dataset:
    imgs/:
        667626_18933d713e.jpg
        ....
    CrowdFlowerAnnotations.txt
    ExpertAnnotations.txt
    Flickr_8k.devImages.txt
    Flickr_8k.testImages.txt
    Flickr_8k.trainImages.txt
    Flickr8k.lemma.token.txt
    Flickr8k.token.txt
    readme.txt
"""


class Flickr8kData(Dataset):
    def __init__(self, root, net_h=224, net_w=224, train=True, transform=None):
        super(Flickr8kData, self).__init__()
        self.root = root
        self.net_h = net_h
        self.net_w = net_w
        self.train = train
        self.transform = transform
        self.train_imgs = []
        self.test_imgs = []
        self.itos = {}
        self.stoi = {}
        self.img_caption = {}
        self.special_tokens = ["<sos>", "<eos>", "<pad>"]
        self.build_vocab()
        self.vocab_size = len(self.itos)
        self.pad_token = self.stoi["<pad>"]
        self.sos_token = self.stoi["<sos>"]
        self.eos_token = self.stoi["<eos>"]
        self.parse_txt()

    def __len__(self):
        if self.train:
            return len(self.train_imgs)
        else:
            return len(self.test_imgs)

    def __getitem__(self, idx):
        if self.train:
            img_id = self.train_imgs[idx].strip()
        else:
            img_id = self.test_imgs[idx].strip()

        img = cv2.imread(os.path.join(self.root, "imgs", img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.net_w, self.net_h))
        description = self.img_caption[img_id][0]
        descp_ind = self.tokenizer(description)
        if self.transform:
            img = self.transform(img)
        return img, descp_ind

    def tokenizer(self, description):
        tokens = description.split()
        descp_ind = []
        # 添加句首标志
        descp_ind.append(self.stoi[self.special_tokens[0]])
        for token in tokens:
            if token in self.stoi:
                descp_ind.append(self.stoi[token])
        # 添加句尾标志
        descp_ind.append(self.stoi[self.special_tokens[1]])

        return descp_ind

    def parse_txt(self):
        self.train_imgs = open(os.path.join(self.root, "Flickr_8k.trainImages.txt"), 'r').readlines()[:1000]
        self.test_imgs = open(os.path.join(self.root, "Flickr_8k.testImages.txt"), 'r').readlines()[:20]

    def build_vocab(self):
        token_file = open(os.path.join(self.root, "Flickr8k.lemma.token.txt"), 'r').read()
        mapping = dict()
        for line in token_file.split('\n'):
            tokens = line.split()
            if len(line) < 2:
                continue
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = image_id.split('#')[0]
            image_desc = ' '.join(image_desc)
            if image_id not in mapping:
                mapping[image_id] = list()
            mapping[image_id].append(image_desc)
        table = str.maketrans('', '', string.punctuation)
        for key, desc_list in mapping.items():
            for i in range(len(desc_list)):
                desc = desc_list[i]
                desc = desc.split()
                desc = [word.lower() for word in desc]
                desc = [w.translate(table) for w in desc]
                # remove hanging 's' and 'a'
                desc = [word for word in desc if len(word) > 1]
                # remove tokens with numbers in them
                desc = [word for word in desc if word.isalpha()]
                # store as string
                desc_list[i] = ' '.join(desc)
        all_desc = set()
        for key in mapping.keys():
            [all_desc.update(d.split()) for d in mapping[key]]
        self.img_caption = mapping
        all_desc = list(all_desc)
        all_desc.sort()
        for idx, item in enumerate(all_desc):
            self.itos[idx] = item
            self.stoi[item] = idx
        # 添加3个特征token, 标志句子开始、结束、填充
        for token in self.special_tokens:
            idx += 1
            self.itos[idx] = token
            self.stoi[token] = idx
