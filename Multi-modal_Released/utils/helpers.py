#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from utils.dataset import JsonlDataset, AddGaussianNoise, AddSaltPepperNoise
from utils.vocab import Vocab


def get_transforms():
    """Get image transforms for training/testing."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_GaussianNoisetransforms(rgb_severity):
    """Get image transforms with Gaussian noise augmentation."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply([AddGaussianNoise(amplitude=rgb_severity * 10)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_SaltNoisetransforms(rgb_severity):
    """Get image transforms with salt-pepper noise augmentation."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply([AddSaltPepperNoise(density=0.1, p=rgb_severity/10)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path):
    """Extract labels and their frequencies from dataset."""
    label_freqs = Counter()
    # print(path)
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    """Extract word list from GloVe embeddings file."""
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    """Initialize vocabulary with BERT tokenizer."""
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab


def collate_fn(batch):
    """Custom collate function for data loading."""
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = torch.stack([row[2] for row in batch])

    tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    idx = torch.cat([row[4] for row in batch]).long() 
    label = torch.cat([row[5] for row in batch]).long()
    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, idx, label


def get_data_loaders(args, shuffle=True):
    """Get data loaders for training, validation, and testing."""
    tokenizer = (BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True).tokenize)

    transforms = get_transforms()

    # Determine training file based on noise ratio
    if args.IDN == 0:
        train_set_filename = "train.jsonl"
    else:
        train_set_filename = f"train_IDN_{args.IDN}.jsonl"

    # Extract labels and frequencies
    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, "food101", train_set_filename)
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    collate = functools.partial(collate_fn)

    # Create datasets
    train = JsonlDataset(
        os.path.join(args.data_path, "food101", train_set_filename),
        tokenizer, transforms, vocab, args,
    )
    train_loader = DataLoader(train,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_workers,collate_fn=collate)

    dev = JsonlDataset(
        os.path.join(args.data_path, "food101", "dev.jsonl"),
        tokenizer, transforms, vocab, args, train=False,
    )

    val_loader = DataLoader(dev,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate)

    test_set = JsonlDataset(
        os.path.join(args.data_path, "food101", "test.jsonl"),
        tokenizer, transforms, vocab, args, train=False,
    )

    test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate)
 
    return train_loader, val_loader, test_loader
