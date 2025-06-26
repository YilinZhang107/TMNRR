#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from utils.util import truncate_seq_pair, numpy_seed

class JsonlDataset(Dataset):
    """Dataset class for loading JSONL format data with images and text."""
    
    def __init__(self, data_path, tokenizer, transforms, vocab, args, train=True):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)

        self.img_dir = "./Multi-modal_Released/datasets/UPMC_Food101/"   # todo
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] 
        self.max_seq_len = 512
        self.transforms = transforms
        self.train = train 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Tokenize text
        tokens = self.tokenizer(self.data[index]["text"])
        sentence = (
            self.text_start_token + tokens[:(self.max_seq_len - 1)]
        )
        segment = torch.zeros(len(sentence))

        # Convert tokens to indices
        sentence = torch.LongTensor([
            self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
            for w in sentence
        ])

        # Get labels
        label = torch.LongTensor([self.args.labels.index(self.data[index]["label"])])

        # Handle noisy labels for training
        if self.args.IDN > 0 and self.train:
            noisy_label = torch.LongTensor([
                self.args.labels.index(self.data[index]["noisy_label"])
            ])
        else:
            noisy_label = label
 
        # Load and transform image
        image = None
        if self.data[index]["img"]:
            img_path = os.path.join(self.img_dir, self.data[index]["img"])
            image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        
        return sentence, segment, image, label, torch.LongTensor([index]), noisy_label

class AddGaussianNoise(object):
    """Add Gaussian noise to images."""
    
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255  # Avoid overflow
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class AddSaltPepperNoise(object):
    """Add salt and pepper noise to images."""
    
    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice(
                (0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd]
            )
            mask = np.repeat(mask, c, axis=2)
            img[mask == 0] = 0    
            img[mask == 1] = 255  
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img